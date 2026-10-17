"""Scout: the cheap, always-on model. Runs every frame, emits "new object at bbox"
only once per object (dedupe + dwell), so the expensive stages fire rarely.
That ratio is the headline number on the dashboard.

Security skin: weapon classes escalate, everything else YOLO names is benign and
gets ignored without a VLM call. People are context, never triggers: a weapon
event carries the nearest person and a head box to aim the evidence zoom at."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from ..events import BUS

BBox = tuple[int, int, int, int]

# COCO classes that escalate. No firearms: COCO has none, and we don't fake it.
WEAPON_CLASSES = frozenset({"knife", "baseball bat", "scissors"})
CONTEXT_CLASSES = frozenset({"person"})
UNKNOWN_LABELS = frozenset({"motion"})  # MotionScout can't name things; investigate the old way


@dataclass
class Detection:
    label: str
    conf: float
    bbox: BBox
    camera: str
    threat: str = "unknown"  # weapon | benign | unknown
    person: BBox | None = None
    head: BBox | None = None


def classify(label: str) -> str:
    if label in WEAPON_CLASSES:
        return "weapon"
    return "unknown" if label in UNKNOWN_LABELS else "benign"


def head_region(person: BBox) -> BBox:
    """Top fifth of a standing person's box, centred. Only used to aim the zoom
    for an evidence crop; nothing downstream identifies or describes the face."""
    x1, y1, x2, y2 = person
    hh = max(int((y2 - y1) * 0.2), 8)
    hw = min(x2 - x1, max(int(hh * 0.9), 8))
    cx = (x1 + x2) // 2
    return (cx - hw // 2, y1, cx + hw // 2, y1 + hh)


def nearest_person(people: list[Detection], bbox: BBox) -> BBox | None:
    """Person box closest to the object's centre (0 if the centre is inside it)."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    def dist(p: Detection) -> float:
        x1, y1, x2, y2 = p.bbox
        return float(np.hypot(max(x1 - cx, 0, cx - x2), max(y1 - cy, 0, cy - y2)))

    return min(people, key=dist).bbox if people else None


class _Tracker:
    """Keeps a short memory of boxes so the same cup isn't 'new' 30 times a second."""

    def __init__(self, iou_thresh: float = 0.4, dwell: float = 0.6, forget: float = 3.0):
        self.seen: dict[int, dict] = {}
        self.next_id = 0
        self.iou_thresh, self.dwell, self.forget = iou_thresh, dwell, forget

    @staticmethod
    def iou(a: BBox, b: BBox) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / area if area else 0.0

    def update(self, dets: list[Detection]) -> list[Detection]:
        """Returns detections that have been stable for `dwell` seconds and not yet reported."""
        now = time.time()
        fresh: list[Detection] = []
        for d in dets:
            match = None
            for tid, rec in self.seen.items():
                if self.iou(rec["bbox"], d.bbox) > self.iou_thresh:
                    match = tid
                    break
            if match is None:
                self.seen[self.next_id] = {"bbox": d.bbox, "first": now, "last": now, "reported": False}
                self.next_id += 1
                continue
            rec = self.seen[match]
            rec["bbox"], rec["last"] = d.bbox, now
            if not rec["reported"] and now - rec["first"] >= self.dwell:
                rec["reported"] = True
                fresh.append(d)
        for tid in [t for t, r in self.seen.items() if now - r["last"] > self.forget]:
            del self.seen[tid]
        return fresh


class YoloScout:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        from ultralytics import YOLO  # imported lazily so mock mode needs no torch
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame: np.ndarray, camera: str) -> list[Detection]:
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append(Detection(res.names[int(b.cls)], float(b.conf), (x1, y1, x2, y2), camera))
        return out


class MotionScout:
    """No-model fallback: background subtraction. Good enough to demo the cascade
    on a machine with no torch, and a legitimate ultra-cheap first stage."""

    def __init__(self, min_area: int = 2500):
        self.sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)
        self.min_area = min_area
        self.primed = False

    def detect(self, frame: np.ndarray, camera: str) -> list[Detection]:
        # First frame becomes the background outright; after that learn slowly so a
        # held-up object stays foreground long enough to be investigated.
        mask = self.sub.apply(frame, learningRate=1.0 if not self.primed else 0.002)
        self.primed = True
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append(Detection("motion", 0.5, (x, y, x + w, y + h), camera))
        return out


class SceneScout:
    """Synthetic-mode detector: reads the scripted scene's ground truth instead of
    running a model (YOLO can't see drawn people). Everything downstream of the
    Scout is the real pipeline."""

    def __init__(self, truth: Callable[[str], list[tuple[str, BBox]]]):
        self.truth = truth

    def detect(self, frame: np.ndarray, camera: str) -> list[Detection]:
        return [Detection(label, 0.9, bbox, camera) for label, bbox in self.truth(camera)]


class Scout:
    def __init__(self, backend: str, model_path: str, dwell: float,
                 truth: Callable[[str], list[tuple[str, BBox]]] | None = None):
        if backend == "scene" and truth is not None:
            self.impl: YoloScout | MotionScout | SceneScout = SceneScout(truth)
        else:
            self.impl = YoloScout(model_path) if backend == "yolo" else MotionScout()
        self.trackers: dict[str, _Tracker] = {}
        self.dwell = dwell
        self.last_dets: dict[str, list[Detection]] = {}  # every box this frame, for the dashboard overlay

    def step(self, frame: np.ndarray, camera: str) -> list[Detection]:
        t0 = time.perf_counter()
        dets = self.impl.detect(frame, camera)
        self.last_dets[camera] = dets
        people = [d for d in dets if d.label in CONTEXT_CLASSES]
        tracker = self.trackers.setdefault(camera, _Tracker(dwell=self.dwell))
        fresh = tracker.update([d for d in dets if d.label not in CONTEXT_CLASSES])
        BUS.bump("frames")
        for d in fresh:
            d.threat = classify(d.label)
            if d.threat == "weapon":
                d.person = nearest_person(people, d.bbox)
                d.head = head_region(d.person) if d.person else None
            BUS.bump("scout_hits")
            near = " (person nearby)" if d.person else ""
            BUS.emit("scout", f"new {d.label} on {camera} at {d.bbox}: {d.threat}{near}",
                     (time.perf_counter() - t0) * 1000, bbox=d.bbox, label=d.label, camera=camera,
                     threat=d.threat, person=d.person, head=d.head)
        return fresh
