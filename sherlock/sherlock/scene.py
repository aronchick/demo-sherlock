"""Scripted synthetic scene for mock mode and B-roll. One loop of the script:
an empty room; someone walks in holding a cup (benign: ignored, no VLM call),
leaves; someone walks in holding a knife (weapon: full alert chain), leaves.

Every camera gets its own ScriptedScene on a shared clock, so all views show
the same moment. `objects()` is the ground truth the SceneScout reads, because
YOLO can't see drawn people. SCENE_SPEED speeds the script up (tests use it)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

BBox = tuple[int, int, int, int]

W, H = 1280, 720
FLOOR_Y = 504
CYCLE = 26.0
OFFSCREEN = 140  # people enter and leave this far outside the frame

SKIN, HAIR, PANTS, SHOES = (120, 150, 190), (38, 38, 44), (52, 48, 46), (28, 28, 30)
OVERLAY = (90, 190, 235)  # amber; deliberately not near-white so it never reads as steel


@dataclass(frozen=True)
class Beat:
    """One person's visit: walk in, stand still holding `item`, walk out."""
    start: float
    stood: float      # walk-in finished
    raised: float     # item visible from here...
    lowered: float    # ...to here, then walk out
    end: float
    stand_x: int
    jacket: tuple[int, int, int]
    item: str         # cup | knife


SCRIPT = (
    Beat(1.0, 3.0, 3.4, 9.0, 11.0, 560, (70, 110, 60), "cup"),
    Beat(12.5, 14.5, 14.9, 21.5, 23.5, 600, (60, 58, 120), "knife"),
)


def _room() -> np.ndarray:
    f = np.empty((H, W, 3), np.uint8)
    shade = np.linspace(0.78, 1.0, FLOOR_Y)[:, None, None]
    f[:FLOOR_Y] = (np.array((64, 57, 50)) * shade).astype(np.uint8)
    f[FLOOR_Y:] = (80, 82, 84)
    for y in (540, 590, 655):
        cv2.line(f, (0, y), (W, y), (72, 74, 76), 2)
    for i in range(-7, 8):
        cv2.line(f, (640 + i * 120, FLOOR_Y), (640 + i * 300, H), (72, 74, 76), 2)
    cv2.rectangle(f, (0, FLOOR_Y - 10), (W, FLOOR_Y), (40, 36, 33), -1)  # baseboard
    cv2.rectangle(f, (100, 160), (260, FLOOR_Y - 10), (30, 28, 26), -1)  # doorway
    cv2.rectangle(f, (100, 160), (260, FLOOR_Y - 10), (92, 86, 80), 8)
    cv2.rectangle(f, (880, 150), (1150, 330), (92, 80, 66), -1)  # window
    cv2.rectangle(f, (880, 150), (1150, 330), (98, 92, 86), 8)
    cv2.line(f, (1015, 150), (1015, 330), (98, 92, 86), 6)
    cv2.line(f, (880, 240), (1150, 240), (98, 92, 86), 6)
    cv2.rectangle(f, (900, 440), (1180, 458), (45, 62, 84), -1)  # bench
    for x in (915, 1160):
        cv2.rectangle(f, (x, 458), (x + 8, 520), (36, 48, 64), -1)
    return f


_ROOM = _room()


def _hand(x: int, holding: bool) -> tuple[int, int]:
    return (x + 84, 368) if holding else (x + 66, 462)


def _draw_person(f: np.ndarray, x: int, jacket: tuple[int, int, int], walking: bool, t: float,
                 holding: bool) -> None:
    sleeve = tuple(int(c * 0.8) for c in jacket)
    swing = int(14 * np.sin(t * 9)) if walking else 0
    for dx, s in ((-1, swing), (1, -swing)):
        lx = x + dx * 25 + s
        cv2.rectangle(f, (lx - 17, 470), (lx + 17, 632), PANTS, -1)
        cv2.rectangle(f, (lx - 21, 628), (lx + 21, 645), SHOES, -1)
    cv2.rectangle(f, (x - 55, 322), (x + 55, 478), jacket, -1)
    cv2.ellipse(f, (x, 324), (55, 20), 0, 180, 360, jacket, -1, cv2.LINE_AA)
    cv2.rectangle(f, (x - 12, 288), (x + 12, 306), SKIN, -1)
    cv2.circle(f, (x, 262), 30, SKIN, -1, cv2.LINE_AA)
    cv2.ellipse(f, (x, 256), (31, 25), 0, 180, 360, HAIR, -1, cv2.LINE_AA)
    cv2.line(f, (x - 64, 330), (x - 66, 452), sleeve, 19)
    cv2.circle(f, (x - 66, 462), 10, SKIN, -1, cv2.LINE_AA)
    if holding:  # bent arm, hand at chest height
        cv2.line(f, (x + 64, 330), (x + 70, 405), sleeve, 19)
        cv2.line(f, (x + 70, 405), (x + 84, 372), sleeve, 17)
    else:
        cv2.line(f, (x + 64, 330), (x + 66, 452), sleeve, 19)
        cv2.circle(f, (x + 66, 462), 10, SKIN, -1, cv2.LINE_AA)


def _draw_item(f: np.ndarray, item: str, hx: int, hy: int) -> BBox:
    if item == "cup":
        red, rim = (55, 75, 190), (90, 110, 220)
        cv2.ellipse(f, (hx + 19, hy - 26), (9, 12), 0, 0, 360, red, 5, cv2.LINE_AA)
        cv2.rectangle(f, (hx - 17, hy - 48), (hx + 17, hy + 4), red, -1)
        cv2.ellipse(f, (hx, hy - 48), (17, 5), 0, 0, 360, rim, -1, cv2.LINE_AA)
        cv2.circle(f, (hx - 4, hy), 11, SKIN, -1, cv2.LINE_AA)
        return (hx - 18, hy - 54, hx + 30, hy + 12)
    blade = np.array([(hx - 7, hy - 14), (hx + 8, hy - 14), (hx + 8, hy - 80),
                      (hx + 2, hy - 112), (hx - 7, hy - 70)], np.int32)
    cv2.fillPoly(f, [blade], (226, 229, 233), cv2.LINE_AA)
    cv2.line(f, (hx + 6, hy - 16), (hx + 6, hy - 78), (205, 208, 212), 1, cv2.LINE_AA)
    cv2.rectangle(f, (hx - 11, hy - 19), (hx + 11, hy - 12), (70, 70, 74), -1)
    cv2.rectangle(f, (hx - 7, hy - 12), (hx + 7, hy + 34), (30, 30, 34), -1)
    cv2.circle(f, (hx, hy + 8), 11, SKIN, -1, cv2.LINE_AA)
    return (hx - 12, hy - 114, hx + 12, hy + 36)


class ScriptedScene:
    """Frame source with the same `read()` contract as the camera sources."""

    def __init__(self, camera: str, t0: float | None = None, speed: float | None = None):
        self.camera = camera
        self.t0 = time.time() if t0 is None else t0
        self.speed = float(os.environ.get("SCENE_SPEED", "1")) if speed is None else speed
        self._objects: list[tuple[str, BBox]] = []

    def clock(self) -> float:
        return ((time.time() - self.t0) * self.speed) % CYCLE

    def objects(self) -> list[tuple[str, BBox]]:
        """Ground truth for the most recent frame: [(label, bbox)]."""
        return list(self._objects)

    def read(self, newer_than: float = 0.0) -> np.ndarray:  # generated frames are always fresh
        t = self.clock()
        f = _ROOM.copy()
        objects: list[tuple[str, BBox]] = []
        for b in SCRIPT:
            if not b.start <= t < b.end:
                continue
            if t < b.stood:
                x = int(-OFFSCREEN + (b.stand_x + OFFSCREEN) * (t - b.start) / (b.stood - b.start))
            elif t < b.lowered:
                x = b.stand_x
            else:
                x = int(b.stand_x + (W + OFFSCREEN - b.stand_x) * (t - b.lowered) / (b.end - b.lowered))
            holding = b.raised <= t < b.lowered
            _draw_person(f, x, b.jacket, not b.stood <= t < b.lowered, t, holding)
            person = (max(x - 78, 0), 230, min(x + 96, W), 646)
            if person[2] - person[0] >= 40:
                objects.append(("person", person))
            if holding:
                objects.append((b.item, _draw_item(f, b.item, *_hand(x, True))))
        self._objects = objects
        self._overlay(f)
        return f

    def _overlay(self, f: np.ndarray) -> None:
        cv2.putText(f, f"CAM {self.camera.upper()}", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, OVERLAY, 2,
                    cv2.LINE_AA)
        cv2.putText(f, time.strftime("%Y-%m-%d  %H:%M:%S"), (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, OVERLAY,
                    1, cv2.LINE_AA)
        if int(time.time() * 2) % 2:
            cv2.circle(f, (W - 70, 34), 8, (50, 50, 220), -1, cv2.LINE_AA)
        cv2.putText(f, "REC", (W - 54, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, OVERLAY, 2, cv2.LINE_AA)
