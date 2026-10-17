"""Main loop. Scout watches every camera's wide view; each fresh detection becomes
one investigation. One investigation at a time so the hero camera isn't fought over."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .agents.director import Director, RuleDirector
from .agents.scout import Scout
from .agents.vision import Analyst, MockAnalyst, MockVerifier, Verifier
from .camera import Camera, CameraError, build_camera
from .config import CONFIG, Config
from .events import BUS
from .tools import ToolBox


class Sherlock:
    def __init__(self, cfg: Config = CONFIG, synthetic: bool = False):
        self.cfg = cfg
        self.cameras: dict[str, Camera] = {
            c.name: build_camera(c.name, c.backend, "synthetic" if synthetic else c.source)
            for c in cfg.cameras}
        if cfg.mock:
            analyst, self.verifier = MockAnalyst(), MockVerifier()
        else:
            analyst, self.verifier = Analyst(cfg.analyst), Verifier(cfg.verifier)
        self.tools = ToolBox(self.cameras, analyst)
        self.director = RuleDirector(self.tools) if cfg.mock else Director(cfg.director, self.tools, cfg.max_tool_calls, cfg.director_thinking)
        self.scout = Scout("mock" if cfg.mock else cfg.scout_backend, cfg.yolo_model, cfg.dwell_seconds)
        self.latest_wide: dict[str, np.ndarray] = {}
        self.latest_crop: np.ndarray | None = None
        self.log: list[dict] = []
        self._busy = threading.Lock()
        self.running = False

    # --- control surface used by the dashboard --------------------------------
    def cover(self, camera: str, covered: bool) -> None:
        self.cameras[camera].covered = covered
        BUS.emit("system", f"{camera} {'covered' if covered else 'uncovered'}")

    def inject_failure(self, n: int = 1) -> None:
        self.tools.inject_failure(n)
        BUS.emit("system", f"next {n} tool call(s) will fail")

    # --- loop -------------------------------------------------------------------
    def start(self) -> None:
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        BUS.emit("system", f"started with cameras {list(self.cameras)}, mock={self.cfg.mock}")

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

    def _investigate(self, det) -> None:
        if not self._busy.acquire(blocking=False):
            BUS.bump("vlm_calls_avoided")
            return
        try:
            t0 = time.perf_counter()
            report = self.director.investigate(det)
            crop = self.tools.last_crop
            if crop is not None:
                self.latest_crop = crop
                verdict = self.verifier.check(crop, report.get("claim", ""))
                report["verified"] = verdict["supported"]
                report["verifier_reason"] = verdict["reason"]
            report["total_ms"] = (time.perf_counter() - t0) * 1000
            report["trigger"] = {"camera": det.camera, "label": det.label, "bbox": det.bbox}
            self.log.append(report)
            BUS.emit("system", f"REPORT: {report.get('claim')} ({report.get('action')}, "
                     f"{'verified' if report.get('verified') else 'unverified'})",
                     report["total_ms"], **{k: v for k, v in report.items() if k != "trigger"})
        finally:
            # avoided calls = frames the Scout handled alone since this investigation began
            self._busy.release()

    def frame_jpeg(self, which: str = "wide") -> bytes | None:
        img = self.latest_crop if which == "crop" else next(iter(self.latest_wide.values()), None)
        if img is None:
            return None
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()
