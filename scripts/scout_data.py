"""Shared vocabulary for the Scout fine-tune: one class map used by capture,
pre-labelling, dataset build, training and eval, so "weapon" means the same thing
at every step and in the runtime Scout (YOLO_MODEL swap).

Labels are YOLO txt: one "cls cx cy w h" line per box, normalised to 0..1."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

# Index = YOLO class id. Person stays a class so weapon events keep a person and a
# head box to aim the evidence zoom at; dropping it would teach the model that
# people are background. Append new classes at the end so existing label ids keep
# their meaning. The fedora and cap are the sponsor props (Red Hat fedora, Red Hat x
# NVIDIA cap): no COCO class covers them, so they come only from in-room photos.
# Coke is a can held in the hand: a benign object the Director ignores on camera,
# kept apart from COCO's bottle so the board names it and cans don't dilute bottle.
CLASSES = ("person", "weapon", "badge", "bottle", "fedora", "cap", "coke")

# Open COCO categories (and stock YOLOv8 names, which are the same) -> Scout class.
# COCO has no badge and no firearm: badges come only from the in-room photos.
COCO_TO_SCOUT = {"person": "person", "knife": "weapon", "scissors": "weapon",
                 "baseball bat": "weapon", "bottle": "bottle"}
COCO_TARGETS = ("knife", "scissors", "baseball bat", "bottle")  # categories we pull images for

# In-room capture sessions. "none" = empty room or a person with no prop: hard negatives.
PROP_CLASS = {"knife": "weapon", "bat": "weapon", "scissors": "weapon",
              "badge": "badge", "bottle": "bottle", "fedora": "fedora", "cap": "cap",
              "coke": "coke", "none": None}
# Stock YOLO has no badge or can class; these are what it usually calls them.
# Pre-label hints only: in that prop's session they become drafts of the prop's class.
PROP_HINTS = {"badge": frozenset({"cell phone", "remote", "book"}),
              "coke": frozenset({"bottle", "cup"})}

# COCO image licences that allow redistribution under CC-BY: 4 Attribution,
# 7 No known copyright restrictions, 8 United States Government Work.
# 1-3 are NonCommercial, 5 ShareAlike, 6 NoDerivs: excluded.
CC_BY_LICENSES = frozenset({4, 7, 8})

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def scout_class(name: str) -> str | None:
    """Map a detector's class name onto the Scout classes; None = not ours."""
    return name if name in CLASSES else COCO_TO_SCOUT.get(name)


# --- YOLO labels ------------------------------------------------------------------
def to_yolo(cls: int, box: BBox, w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
    y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def from_yolo(line: str, w: int, h: int) -> tuple[int, BBox]:
    c, cx, cy, bw, bh = line.split()
    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
    return int(c), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def read_labels(path: Path, w: int, h: int) -> list[tuple[int, BBox]]:
    if not path.exists():
        return []
    return [from_yolo(ln, w, h) for ln in path.read_text().splitlines() if ln.strip()]


def write_labels(path: Path, labels: Iterable[tuple[int, BBox]], w: int, h: int) -> None:
    lines = [to_yolo(c, b, w, h) for c, b in labels if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
    path.write_text("".join(f"{ln}\n" for ln in lines))


def coco_xyxy(bbox: list[float]) -> BBox:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


# --- splits -------------------------------------------------------------------------
def hash_split(key: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> str:
    """Stable train/val/test for independent images (COCO): same key, same split, any machine."""
    u = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if u < ratios[0] else "val" if u < ratios[0] + ratios[1] else "test"


def room_splits(paths: Iterable[Path], ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)) -> dict[Path, str]:
    """In-room frames come in bursts, so neighbours are near-duplicates and a random
    split leaks the test set into training. Split each session by time instead:
    the first 70% train, the next 10% val, the last 20% test."""
    sessions: dict[Path, list[Path]] = defaultdict(list)
    for p in paths:
        sessions[p.parent].append(p)
    out: dict[Path, str] = {}
    for frames in sessions.values():
        frames.sort()
        n = len(frames)
        n_train = round(n * ratios[0])
        n_val = round(n * (ratios[0] + ratios[1])) - n_train
        for i, p in enumerate(frames):
            out[p] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


def room_images(root: Path) -> list[Path]:
    """data/props/<prop>/<session>/NNNN.jpg"""
    return sorted(p for p in root.glob("*/*/*") if p.suffix.lower() in IMAGE_EXTS)


def prop_of(image: Path) -> str:
    return image.parent.parent.name


REVIEWED = "reviewed.txt"  # in data/props: photos whose labels a human has checked


def load_reviewed(props: Path) -> set[str]:
    f = props / REVIEWED
    return set(f.read_text().split("\n")) - {""} if f.exists() else set()


# --- metrics ------------------------------------------------------------------------
def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def match_counts(preds: list[tuple[str, float, BBox]], gts: list[tuple[str, BBox]],
                 iou_thr: float = 0.5) -> dict[str, dict[str, int]]:
    """Greedy, highest confidence first, one prediction per ground-truth box,
    same class only. Returns {class: {tp, fp, fn}} for one image."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    used: set[int] = set()
    for cls, _, box in sorted(preds, key=lambda p: -p[1]):
        best, best_i = iou_thr, None
        for i, (gcls, gbox) in enumerate(gts):
            if i in used or gcls != cls:
                continue
            o = iou(box, gbox)
            if o >= best:
                best, best_i = o, i
        if best_i is None:
            counts[cls]["fp"] += 1
        else:
            used.add(best_i)
            counts[cls]["tp"] += 1
    for i, (gcls, _) in enumerate(gts):
        if i not in used:
            counts[gcls]["fn"] += 1
    return counts


def precision_recall(c: dict[str, int]) -> tuple[float, float]:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return (tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0)
