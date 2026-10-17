"""Faces out of every crop a model sees.

The aim rules keep a head zoom from reaching a model; this is the second line. A small face
detector finds any face still inside a weapon or object crop (a knife held at a seated
child's mouth, someone behind the cup) and fills it solid before the crop goes to the
Analyst or the Verifier. Detection only: boxes, no landmarks kept, no embedding, no
recognition. The detector is YuNet (opencv_zoo, MIT, 232 KB, see models/LICENSE-yunet) run
by OpenCV's own FaceDetectorYN; OpenCV 5 ships no Haar cascades. A detector does not count
toward the 33B model budget."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

YUNET = Path(__file__).with_name("models") / "face_detection_yunet_2023mar.onnx"
# Each crop is scaled to this long side for detection, up or down. Weapon crops can be 100 px
# across, and YuNet's smallest anchor is 10 px of its input.
DETECT_SIDE = 640
# The face a box-plus-margin crop can still hold is one cut by its edge: a knife at a mouth.
# Mirroring the edge outward completes it. On COCO val2017 (2026-10-17) the mouth and chin
# at the top of 000000394940's old crop scored 0.49 plain and 0.89 mirrored; the chin above
# 000000048396's upright knife went from no detection to 0.73. The cost seen: one grey patch
# on a dark gap beside 000000199771's jug (0.62).
EDGE_PAD = 0.25
# Low on purpose: a missed face reaches a model, a false one greys out a patch of the crop.
SCORE = 0.5
# YuNet's box runs brow to chin; grow every side so the hairline, ears and jaw go too.
FACE_MARGIN = 0.25
FILL = (128, 128, 128)  # flat grey: not skin, not steel, nothing for a VLM to read

Box = tuple[float, float, float, float]


def fill(img: np.ndarray, boxes: list[Box], margin: float = FACE_MARGIN) -> np.ndarray:
    """A copy of img with each box, plus `margin` of its size per side, filled solid."""
    out = img.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        out[max(int(y1 - my), 0):min(int(np.ceil(y2 + my)), h),
            max(int(x1 - mx), 0):min(int(np.ceil(x2 + mx)), w)] = FILL
    return out


class FaceMasker:
    """Call it on a model-bound crop: returns (masked copy, faces found). One per ToolBox,
    since the OpenCV detector keeps its input size between calls."""

    def __init__(self, model: Path = YUNET, score: float = SCORE):
        if not model.is_file():
            raise FileNotFoundError(f"face detector missing: {model}; no crop may reach a model without it")
        self._net = cv2.FaceDetectorYN.create(str(model), "", (DETECT_SIDE, DETECT_SIDE), score, 0.3, 50)

    def faces(self, img: np.ndarray) -> list[Box]:
        """Face boxes in img's own pixels; they can run past its edges."""
        p = int(EDGE_PAD * max(img.shape[:2]))
        padded = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REFLECT_101)
        h, w = padded.shape[:2]
        s = DETECT_SIDE / max(h, w)
        scaled = cv2.resize(padded, (max(round(w * s), 1), max(round(h * s), 1)))
        self._net.setInputSize((scaled.shape[1], scaled.shape[0]))
        _, found = self._net.detect(scaled)
        if found is None:
            return []
        return [(x / s - p, y / s - p, (x + bw) / s - p, (y + bh) / s - p)
                for x, y, bw, bh in found[:, :4].tolist()]

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, int]:
        boxes = self.faces(img)
        return (fill(img, boxes), len(boxes)) if boxes else (img, 0)
