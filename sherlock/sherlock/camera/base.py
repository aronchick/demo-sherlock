"""Camera abstraction. The Director only ever sees this interface, so a flat webcam
with software crop/zoom and a motorized OBSBOT are interchangeable."""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised for real failures (lens covered, device gone). The Director is
    expected to recover from these, so they must be loud, not silent."""


@dataclass
class Pose:
    pan: float = 0.0   # degrees, -140..140
    tilt: float = 0.0  # degrees, -70..30
    zoom: float = 1.0  # 1..4


class Camera(ABC):
    name: str
    pose: Pose
    covered: bool = False  # error-injection switch for the demo

    @abstractmethod
    def wide_frame(self) -> np.ndarray:
        """Full field of view, used by the Scout."""

    @abstractmethod
    def snapshot(self) -> np.ndarray:
        """What the camera sees at its current pose, used by the Analyst."""

    @abstractmethod
    def move(self, pan: float | None = None, tilt: float | None = None,
             zoom: float | None = None) -> Pose:
        ...

    def _check(self) -> None:
        if self.covered:
            raise CameraError(f"{self.name}: no usable image (lens covered or stream dead)")

    def describe(self) -> dict:
        return {"name": self.name, "pan": self.pose.pan, "tilt": self.pose.tilt,
                "zoom": self.pose.zoom, "covered": self.covered}


class _FrameSource:
    """Background reader so slow consumers never see a stale buffer."""

    def __init__(self, source: str):
        src = int(source) if source.isdigit() else source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise CameraError(f"cannot open video source {source!r}")
        self.frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                # loop video files for demos
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f

    def read(self) -> np.ndarray:
        for _ in range(100):
            with self.lock:
                if self.frame is not None:
                    return self.frame.copy()
            time.sleep(0.02)
        raise CameraError("no frames from source")


class SyntheticSource:
    """Generates frames with a labelled object drifting across a room. Lets the
    whole pipeline run with no hardware at all (CI, sandbox, airplane)."""

    def __init__(self, w: int = 1280, h: int = 720):
        self.w, self.h, self.t0 = w, h, time.time()

    def read(self) -> np.ndarray:
        t = time.time() - self.t0
        f = np.full((self.h, self.w, 3), 40, np.uint8)
        cv2.rectangle(f, (0, int(self.h * 0.7)), (self.w, self.h), (70, 70, 70), -1)
        # someone holds up an object at t=1s, keeps it still, drops it at t=12s
        if 1 < t % 20 < 12:
            x, y = int(self.w * 0.68), int(self.h * 0.52)
            cv2.rectangle(f, (x, y), (x + 90, y + 140), (200, 90, 30), -1)  # BGR: blue
            cv2.putText(f, "ACME", (x + 8, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(f, "COLA", (x + 8, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return f


class FakePTZCamera(Camera):
    """Software pan/tilt/zoom: crops the wide frame. Pan/tilt map linearly onto
    the frame so the Director's tool calls carry the same units as the OBSBOT."""

    PAN_RANGE, TILT_RANGE = 140.0, 70.0

    def __init__(self, name: str, source: str | None):
        self.name = name
        self.pose = Pose()
        self.src = SyntheticSource() if source in (None, "synthetic") else _FrameSource(source)

    def wide_frame(self) -> np.ndarray:
        self._check()
        return self.src.read()

    def snapshot(self) -> np.ndarray:
        f = self.wide_frame()
        h, w = f.shape[:2]
        z = self.pose.zoom
        cw, ch = int(w / z), int(h / z)
        cx = int(w / 2 + (self.pose.pan / self.PAN_RANGE) * (w / 2))
        cy = int(h / 2 - (self.pose.tilt / self.TILT_RANGE) * (h / 2))
        x0 = min(max(cx - cw // 2, 0), w - cw)
        y0 = min(max(cy - ch // 2, 0), h - ch)
        crop = f[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def move(self, pan=None, tilt=None, zoom=None) -> Pose:
        self._check()
        if pan is not None:
            self.pose.pan = float(max(-self.PAN_RANGE, min(self.PAN_RANGE, pan)))
        if tilt is not None:
            self.pose.tilt = float(max(-self.TILT_RANGE, min(30.0, tilt)))
        if zoom is not None:
            self.pose.zoom = float(max(1.0, min(4.0, zoom)))
        return self.pose

    @staticmethod
    def bbox_to_pose(bbox: tuple[int, int, int, int], frame_wh: tuple[int, int]) -> Pose:
        """Where to point so a wide-frame bbox fills the view. Used by the
        `look_at` tool so the Director reasons in boxes, not degrees."""
        x1, y1, x2, y2 = bbox
        w, h = frame_wh
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        pan = (cx - w / 2) / (w / 2) * FakePTZCamera.PAN_RANGE
        tilt = -(cy - h / 2) / (h / 2) * FakePTZCamera.TILT_RANGE
        zoom = max(1.0, min(4.0, 0.8 * min(w / max(x2 - x1, 1), h / max(y2 - y1, 1))))
        return Pose(pan, tilt, zoom)
