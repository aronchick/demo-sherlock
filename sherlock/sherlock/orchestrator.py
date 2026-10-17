"""Main loop. Scout watches every camera's wide view; each fresh detection becomes
one investigation. One investigation at a time so the hero camera isn't fought over."""
from __future__ import annotations

import os
import pathlib
import threading
import time

import cv2
import numpy as np

from .agents.director import Director, RuleDirector, weapon_claim
from .agents.scout import Detection, Scout, classify
from .agents.vision import Analyst, MockAnalyst, MockVerifier, Verifier
from .camera import Camera, CameraError, build_camera
from .config import CONFIG, Config
from .edge import EdgeSink
from .events import BUS
from .incidents import IncidentStore
from .scene import ScriptedScene
from .tools import ToolBox

# Cameras share a room, so the same knife shows up on each. One incident per
# object: a second camera reporting the same label this soon is a duplicate.
DEDUPE_SECONDS = 8.0

_BOX_COLORS = {"weapon": (70, 70, 240), "benign": (130, 200, 140), "unknown": (90, 190, 235),
               "person": (200, 170, 120)}


class Sherlock:
    def __init__(self, cfg: Config = CONFIG, synthetic: bool = False, evidence_dir: str | None = None):
        self.cfg = cfg
        # A camera that will not open (unplugged, wrong index or name) is left out,
        # loudly, so the demo carries on with the others. With none, stop.
        self.cameras: dict[str, Camera] = {}
        self.unavailable: dict[str, str] = {}  # camera -> why it did not open
        for c in cfg.cameras:
            try:
                self.cameras[c.name] = build_camera(c.name, c.backend, "synthetic" if synthetic else c.source)
            except CameraError as e:
                self.unavailable[c.name] = str(e)
                BUS.emit("system", f"{c.name} unavailable: {e}; continuing without it")
        if self.unavailable and not self.cameras:
            why = "; ".join(f"{name}: {err}" for name, err in self.unavailable.items())
            raise CameraError(f"no camera opened ({why}); check CAMERAS")
        self.scenes: dict[str, ScriptedScene] = {}
        if synthetic:
            t0 = time.time()
            for name, cam in self.cameras.items():
                if hasattr(cam, "src"):  # software-PTZ cameras crop whatever .src returns
                    cam.src = self.scenes[name] = ScriptedScene(name, t0)
        if cfg.mock:
            analyst, self.verifier = MockAnalyst(), MockVerifier()
        else:
            analyst, self.verifier = Analyst(cfg.analyst), Verifier(cfg.verifier)
        self.tools = ToolBox(self.cameras, analyst, verifier=self.verifier)
        self.director = RuleDirector(self.tools) if cfg.mock else Director(cfg.director, self.tools, cfg.max_tool_calls, cfg.director_thinking)
        backend = "scene" if self.scenes else ("mock" if cfg.mock else cfg.scout_backend)
        # Scout latency is measured only when a real detector runs in the live runtime.
        self.scout = Scout(backend, cfg.yolo_model, cfg.dwell_seconds, truth=self._scene_truth,
                           measure=not cfg.mock and backend != "scene", handled_via=self._handled_via,
                           conf=cfg.scout_conf)
        self.evidence_dir = pathlib.Path(evidence_dir or os.environ.get("EVIDENCE_DIR", "snapshots"))
        self.incidents = IncidentStore(self.evidence_dir)
        self.edge = EdgeSink.from_env(self.evidence_dir)  # None unless SHERLOCK_EDGE=1
        self.latest_wide_at: dict[str, float] = {}
        self.latest_wide: dict[str, np.ndarray] = {}
        self.log: list[dict] = []
        self.active: Detection | None = None
        self._recent: dict[str, tuple[str, float]] = {}  # label -> (camera, when)
        self._incidents = 0
        self._busy = threading.Lock()
        self._thread: threading.Thread | None = None
        self.running = False

    def _scene_truth(self, camera: str) -> list:
        scene = self.scenes.get(camera)
        return scene.objects() if scene else []

    # --- control surface used by the dashboard --------------------------------
    def cover(self, camera: str, covered: bool) -> bool:
        if camera not in self.cameras:  # unavailable or unknown: nothing to cover
            return False
        self.cameras[camera].covered = covered
        BUS.emit("system", f"{camera} {'covered' if covered else 'uncovered'}")
        return True

    def inject_failure(self, n: int = 1) -> None:
        self.tools.inject_failure(n)
        BUS.emit("system", f"next {n} tool call(s) will fail")

    def inject_corruption(self, n: int = 1) -> None:
        self.tools.inject_corruption(n)
        BUS.emit("system", f"next {n} tool call(s) will get corrupted arguments")

    # --- loop -------------------------------------------------------------------
    def start(self) -> None:
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self.edge:
            self.edge.start()
        missing = f", unavailable {list(self.unavailable)}" if self.unavailable else ""
        BUS.emit("system", f"started with cameras {list(self.cameras)}{missing}, mock={self.cfg.mock}"
                 + (", Expanso Edge sink on" if self.edge else ""))

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=10)
        if self.edge:
            self.edge.stop()

    def _loop(self) -> None:
        down: set[str] = set()  # loud once per outage, not every frame: keeps the tool chain readable
        # A failed camera is retried after a pause instead of sleeping the loop, so
        # one covered lens does not slow every other camera to two frames a second.
        retry_at: dict[str, float] = {}
        while self.running:
            for name, cam in self.cameras.items():
                if time.monotonic() < retry_at.get(name, 0.0):
                    continue
                try:
                    frame = cam.wide_frame()
                except CameraError as e:
                    if name not in down:
                        down.add(name)
                        BUS.emit("system", str(e))
                    retry_at[name] = time.monotonic() + 0.5
                    continue
                if name in down:
                    down.discard(name)
                    BUS.emit("system", f"{name}: back online")
                self.latest_wide[name] = frame
                self.latest_wide_at[name] = time.time()
                for det in self.scout.step(frame, name):
                    self._investigate(det)
            time.sleep(0.03)

    def _handled_via(self, det: Detection) -> str | None:
        """The other camera that handled this object moments ago, if any."""
        seen = self._recent.get(det.label)
        if seen and seen[0] != det.camera and time.monotonic() - seen[1] < DEDUPE_SECONDS:
            return seen[0]
        return None

    def _duplicate(self, det: Detection) -> bool:
        via = self._handled_via(det)
        if via:
            BUS.bump("duplicates")
            BUS.emit("system", f"{det.label} on {det.camera} already handled via {via}")
            return True
        return False

    def _investigate(self, det: Detection) -> None:
        if self._duplicate(det):
            return
        if not self._busy.acquire(blocking=False):
            BUS.bump("dropped_busy")  # dropped, not 'avoided': nobody looked at it
            return
        try:
            self._recent[det.label] = (det.camera, time.monotonic())
            self._incidents += 1
            iid = f"{time.strftime('%Y%m%d-%H%M%S')}-{self._incidents:03d}"
            while (self.evidence_dir / iid).exists():
                self._incidents += 1
                iid = f"{time.strftime('%Y%m%d-%H%M%S')}-{self._incidents:03d}"
            self.tools.begin_incident(det, iid)
            self.active = det
            vlm_before = BUS.stats["vlm_calls"]
            t0 = time.perf_counter()
            with BUS.capture() as custody:
                custody.append({"stage": "scout", "t": time.time(),
                                "detail": f"Investigation opened: {det.label} on {det.camera}"})
                self.tools.defer_analysis = True
                saved, unsaved = [], []

                def persist():
                    nonlocal saved, unsaved
                    saved, unsaved = self._save_evidence(iid, self.tools.evidence)
                    pending = {"incident": iid, "t": time.time(), "action": "log",
                               "claim": weapon_claim(det.label), "verified": False,
                               "state": "investigating", "evidence": saved,
                               "evidence_unsaved": unsaved,
                               "evidence_provenance": self.tools.provenance.copy()}
                    try:
                        self.incidents.save(pending, custody, {
                            "mock": self.cfg.mock, "scout": self.cfg.yolo_model,
                            "director": self.cfg.director.model, "analyst": self.cfg.analyst.model,
                            "verifier": self.cfg.verifier.model})
                    except (OSError, ValueError) as exc:
                        BUS.emit("system", f"Pending incident manifest NOT saved for {iid}: {exc}")

                self.tools.on_capture = persist
                source = self.latest_wide.get(det.camera)
                captured = self.latest_wide_at.get(det.camera)
                if source is None:
                    try:
                        source = self.cameras[det.camera].wide_frame()
                        captured = time.time()
                    except CameraError as exc:
                        BUS.emit("system", f"Original frame unavailable: {exc}")
                if source is not None:
                    self.tools.evidence["original_frame"] = source.copy()
                    self.tools.evidence_camera["original_frame"] = det.camera
                    self.tools.evidence_captured_at["original_frame"] = captured
                    self.tools.provenance["original_frame"] = {
                        "camera": det.camera, "captured_at": captured,
                        "source_artifact": None, "relationship": "source_frame", "bbox": None}
                    if det.head:
                        height, width = source.shape[:2]
                        x1, y1, x2, y2 = det.head
                        x1, x2 = max(0, int(x1)), min(width, int(x2))
                        y1, y2 = max(0, int(y1)), min(height, int(y2))
                        if x2 > x1 and y2 > y1:
                            self.tools.evidence["face_crop"] = source[y1:y2, x1:x2].copy()
                            self.tools.evidence_camera["face_crop"] = det.camera
                            self.tools.evidence_captured_at["face_crop"] = captured
                            self.tools.provenance["face_crop"] = {
                                "camera": det.camera, "captured_at": captured,
                                "source_artifact": "original_frame", "relationship": "crop_of_source",
                                "bbox": [x1, y1, x2, y2]}
                    persist()
                try:
                    report = self.director.investigate(det)
                except Exception as exc:  # noqa: BLE001 - preserve captured evidence on an interrupted investigation
                    report = {"action": "log", "claim": weapon_claim(det.label),
                              "confidence": 0, "error": type(exc).__name__}
                    BUS.emit("system", f"Investigation failed: {type(exc).__name__}")
                if det.threat != "benign":
                    claim = weapon_claim(det.label)
                    comparison = self.tools.assess_parallel(claim)
                    report["claim"] = claim
                    report["comparison"] = comparison
                    report["verified"] = (not report.get("error") and comparison["analyst_supports"] is True and
                                          comparison["verifier_supports"] is True)
                    report["action"] = "alert" if det.threat == "weapon" and report["verified"] else "log"
                    report["verifier_reason"] = (self.tools.verdict or {}).get("reason", "")
                    eid, _ = self.tools.model_crop()
                    report["description"] = self.tools.descriptions.get(eid, "")
                report["incident"] = iid
                report["t"] = time.time()
                report["evidence"] = saved
                report["evidence_provenance"] = self.tools.provenance.copy()
                if unsaved:
                    report["evidence_unsaved"] = unsaved
                if report["evidence"]:
                    primary = next((n for n in ("weapon_crop", "object_crop") if n in report["evidence"]),
                                   report["evidence"][0])
                    report["evidence_camera"] = self.tools.evidence_camera[primary]
                report["vlm_calls"] = BUS.stats["vlm_calls"] - vlm_before
                report["total_ms"] = (time.perf_counter() - t0) * 1000
                report["trigger"] = {"camera": det.camera, "label": det.label, "bbox": det.bbox,
                                     "threat": det.threat, "person": det.person, "head": det.head}
            action = report.get("action")
            BUS.bump({"alert": "alerts", "ignore": "ignored"}.get(action, "logged"))
            try:
                record = self.incidents.save(report, custody, {
                    "mock": self.cfg.mock, "scout": self.cfg.yolo_model,
                    "director": self.cfg.director.model, "analyst": self.cfg.analyst.model,
                    "verifier": self.cfg.verifier.model,
                })
                report["notification"] = record["notification"]
                report["custody_saved"] = True
            except (OSError, ValueError) as exc:
                report["custody_saved"] = False
                report["notification"] = {"simulated": True, "status": "failed",
                                          "delivered_at": None, "acknowledged_at": None}
                BUS.emit("system", f"Incident manifest NOT saved for {iid}: {exc}")
            self.log.append(report)
            verdict = "" if "verified" not in report else (", verified" if report["verified"] else ", NOT verified")
            BUS.emit("system", f"REPORT: {report.get('claim')} ({action}{verdict}, "
                     f"{report['vlm_calls']} VLM calls)", report["total_ms"],
                     label=det.label, threat=det.threat, camera=det.camera,
                     **{k: v for k, v in report.items() if k != "trigger"})
        finally:
            self.tools.on_capture = None
            self.tools.defer_analysis = False
            self.active = None
            self._busy.release()

    def _save_evidence(self, iid: str, evidence: dict[str, np.ndarray]) -> tuple[list[str], list[str]]:
        """One folder per incident: <evidence_dir>/<incident>/<weapon_crop|face_crop|object_crop>.jpg.
        Returns (saved, unsaved). A full or read-only disk loses the crops, never the
        report, and never the Scout loop."""
        if not evidence:
            return [], []
        folder = self.evidence_dir / iid
        saved: list[str] = []
        try:
            folder.mkdir(parents=True, exist_ok=True)
            for name, img in evidence.items():
                suffix = ".png" if name == "model_context" else ".jpg"
                if cv2.imwrite(str(folder / f"{name}{suffix}"), img):
                    saved.append(name)
        except (OSError, cv2.error) as e:
            BUS.emit("system", f"evidence write failed in {folder}: {e}")
        unsaved = [n for n in evidence if n not in saved]
        if unsaved:
            BUS.bump("evidence_unsaved")
            BUS.emit("system", f"evidence NOT saved for {iid}: {', '.join(unsaved)}")
        return saved, unsaved

    def evidence_path(self, iid: str, name: str) -> pathlib.Path:
        suffix = ".png" if name == "model_context" else ".jpg"
        return self.incidents.path(iid, f"{name}{suffix}")

    # --- dashboard frames -------------------------------------------------------
    def _annotate(self, camera: str, frame: np.ndarray) -> np.ndarray:
        """Scout boxes on the wide view, plus the face-crop aim during a weapon incident."""
        f = frame.copy()
        boxes = [("person" if d.label == "person" else classify(d.label), d.label, d.bbox)
                 for d in self.scout.last_dets.get(camera, [])]
        det = self.active
        if det is not None and det.head and det.camera == camera:
            boxes.append(("unknown", "face crop", det.head))
        for kind, label, (x1, y1, x2, y2) in boxes:
            color = _BOX_COLORS[kind]
            cv2.rectangle(f, (x1, y1), (x2, y2), color, 3 if kind == "weapon" else 2)
            text = label.upper() if kind == "weapon" else label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(f, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(f, text, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (24, 24, 24), 1, cv2.LINE_AA)
        return f

    def frame_jpeg(self, which: str = "wide", camera: str | None = None) -> bytes | None:
        """JPEG for the dashboard: the latest crop, or one camera's wide view
        (the first camera when none is named)."""
        if which == "crop":
            img = self.tools.last_crop  # live: follows the chain from weapon crop to face crop
        else:
            name = camera or next(iter(self.latest_wide), "")
            img = self.latest_wide.get(name)
            if img is not None and self.cameras[name].covered:
                img = (img * 0.15).astype(np.uint8)  # last frame is stale: say so instead of freezing
                cv2.putText(img, f"{name.upper()}: LENS COVERED", (40, img.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            1.4, (90, 190, 235), 3, cv2.LINE_AA)
            elif img is not None:
                img = self._annotate(name, img)
        if img is None:
            return None
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()
