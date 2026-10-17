"""Main loop. Scout watches every camera's wide view; each fresh detection becomes
one investigation. One investigation at a time so the hero camera isn't fought over."""
from __future__ import annotations

import os
import pathlib
import threading
import time

import cv2
import numpy as np

from .agents.director import Director, RuleDirector
from .agents.scout import Detection, Scout, classify
from .agents.vision import Analyst, MockAnalyst, MockVerifier, Verifier
from .camera import Camera, CameraError, build_camera
from .config import CONFIG, Config
from .events import BUS
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
        self.cameras: dict[str, Camera] = {
            c.name: build_camera(c.name, c.backend, "synthetic" if synthetic else c.source)
            for c in cfg.cameras}
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
        self.scout = Scout(backend, cfg.yolo_model, cfg.dwell_seconds, truth=self._scene_truth)
        self.evidence_dir = pathlib.Path(evidence_dir or os.environ.get("EVIDENCE_DIR", "snapshots"))
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
    def cover(self, camera: str, covered: bool) -> None:
        self.cameras[camera].covered = covered
        BUS.emit("system", f"{camera} {'covered' if covered else 'uncovered'}")

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
        BUS.emit("system", f"started with cameras {list(self.cameras)}, mock={self.cfg.mock}")

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        while self.running:
            for name, cam in self.cameras.items():
                try:
                    frame = cam.wide_frame()
                except CameraError as e:
                    BUS.emit("system", str(e))
                    time.sleep(0.5)
                    continue
                self.latest_wide[name] = frame
                for det in self.scout.step(frame, name):
                    self._investigate(det)
            time.sleep(0.03)

    def _duplicate(self, det: Detection) -> bool:
        seen = self._recent.get(det.label)
        if seen and seen[0] != det.camera and time.monotonic() - seen[1] < DEDUPE_SECONDS:
            BUS.bump("duplicates")
            BUS.emit("system", f"{det.label} on {det.camera} already handled via {seen[0]}")
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
            self.tools.begin_incident(det, iid)
            self.active = det
            vlm_before = BUS.stats["vlm_calls"]
            t0 = time.perf_counter()
            report = self.director.investigate(det)
            ev = self.tools.evidence
            crop = ev.get("weapon_crop", ev.get("object_crop"))
            if self.tools.verdict is None and crop is not None and report.get("action") != "ignore":
                # Safety net for an LLM Director that skipped verify: weapon crop only, never the face.
                self.tools.verdict = self.verifier.check(crop, report.get("claim", ""))
            if self.tools.verdict is not None:
                report["verified"] = bool(self.tools.verdict["supported"])
                report["verifier_reason"] = self.tools.verdict["reason"]
            report["incident"] = iid
            report["evidence"] = self._save_evidence(iid, ev)
            report["vlm_calls"] = BUS.stats["vlm_calls"] - vlm_before
            report["total_ms"] = (time.perf_counter() - t0) * 1000
            report["trigger"] = {"camera": det.camera, "label": det.label, "bbox": det.bbox,
                                 "threat": det.threat, "person": det.person, "head": det.head}
            action = report.get("action")
            BUS.bump({"alert": "alerts", "ignore": "ignored"}.get(action, "logged"))
            self.log.append(report)
            verdict = "" if "verified" not in report else (", verified" if report["verified"] else ", NOT verified")
            BUS.emit("system", f"REPORT: {report.get('claim')} ({action}{verdict}, "
                     f"{report['vlm_calls']} VLM calls)", report["total_ms"],
                     label=det.label, threat=det.threat, camera=det.camera,
                     **{k: v for k, v in report.items() if k != "trigger"})
        finally:
            self.active = None
            self._busy.release()

    def _save_evidence(self, iid: str, evidence: dict[str, np.ndarray]) -> list[str]:
        """One folder per incident: <evidence_dir>/<incident>/<weapon_crop|face_crop|object_crop>.jpg"""
        if not evidence:
            return []
        folder = self.evidence_dir / iid
        folder.mkdir(parents=True, exist_ok=True)
        for name, img in evidence.items():
            cv2.imwrite(str(folder / f"{name}.jpg"), img)
        return list(evidence)

    def evidence_path(self, iid: str, name: str) -> pathlib.Path:
        return self.evidence_dir / iid / f"{name}.jpg"

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

    def frame_jpeg(self, which: str = "wide") -> bytes | None:
        if which == "crop":
            img = self.tools.last_crop  # live: follows the chain from weapon crop to face crop
        else:
            name, img = next(iter(self.latest_wide.items()), ("", None))
            if img is not None:
                img = self._annotate(name, img)
        if img is None:
            return None
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()
