"""Fix the stock-YOLO draft labels on the in-room photos, one keypress per photo.

    python scripts/review_labels.py --split test     # the held-out photos first: they grade the eval
    python scripts/review_labels.py                  # then everything not yet reviewed
    python scripts/review_labels.py --status         # counts only, no window

Mouse: drag = new box in the current class, right-click = delete the box under
the pointer. Keys: 1-6 pick the class (1 person, 2 weapon, 3 badge, 4 bottle, 5 fedora, 6 cap),
r = give the box under the pointer the current class, u = undo, x = clear all,
n / space / enter = save and next, p = save and back, q / esc = save and quit.
Saving marks the photo reviewed (data/props/reviewed.txt); prelabel never
overwrites a reviewed photo."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from scout_data import (
    CLASSES,
    PROP_CLASS,
    REVIEWED,
    load_reviewed,
    read_labels,
    room_images,
    room_splits,
    write_labels,
)

ROOT = Path(__file__).resolve().parents[1]
COLORS = [(200, 170, 120), (70, 70, 240), (60, 200, 240), (130, 200, 140),
          (60, 60, 200), (200, 200, 200)]  # BGR per class
MIN_SIDE = 4  # pixels; smaller drags are mis-clicks


class LabelEditor:
    """Box edits for one photo. No window code, so it can be tested headless."""

    def __init__(self, labels: list[tuple[int, tuple]], cls: str = "weapon"):
        self.labels = list(labels)
        self.cls = CLASSES.index(cls)
        self._history: list[list[tuple[int, tuple]]] = []

    def _push(self) -> None:
        self._history.append(list(self.labels))

    def _hit(self, x: float, y: float) -> int | None:
        """Index of the smallest box containing the point: nested boxes stay reachable."""
        inside = [(abs((b[2] - b[0]) * (b[3] - b[1])), i) for i, (_, b) in enumerate(self.labels)
                  if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
        return min(inside)[1] if inside else None

    def add(self, p0: tuple[float, float], p1: tuple[float, float]) -> bool:
        x1, x2 = sorted((p0[0], p1[0]))
        y1, y2 = sorted((p0[1], p1[1]))
        if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
            return False
        self._push()
        self.labels.append((self.cls, (x1, y1, x2, y2)))
        return True

    def delete_at(self, x: float, y: float) -> bool:
        i = self._hit(x, y)
        if i is None:
            return False
        self._push()
        del self.labels[i]
        return True

    def relabel_at(self, x: float, y: float) -> bool:
        i = self._hit(x, y)
        if i is None:
            return False
        self._push()
        self.labels[i] = (self.cls, self.labels[i][1])
        return True

    def clear(self) -> None:
        self._push()
        self.labels = []

    def undo(self) -> None:
        if self._history:
            self.labels = self._history.pop()


def mark_reviewed(props: Path, rel: str) -> None:
    done = load_reviewed(props) | {rel}
    (props / REVIEWED).write_text("".join(f"{r}\n" for r in sorted(done)))


def pick(props: Path, split: str, only_unreviewed: bool) -> list[Path]:
    photos = room_images(props)
    splits = room_splits(photos)
    done = load_reviewed(props)
    return [p for p in photos if (split == "all" or splits[p] == split)
            and not (only_unreviewed and str(p.relative_to(props)) in done)]


def _draw(img, ed: LabelEditor, scale: float, drag, header: str):
    view = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1 else img.copy()
    for c, (x1, y1, x2, y2) in ed.labels:
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(view, p1, p2, COLORS[c], 2)
        cv2.putText(view, CLASSES[c], (p1[0] + 3, p1[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[c], 1,
                    cv2.LINE_AA)
    if drag:
        cv2.rectangle(view, drag[0], drag[1], COLORS[ed.cls], 1)
    cv2.rectangle(view, (0, 0), (view.shape[1], 26), (24, 24, 24), -1)
    cv2.putText(view, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return view


def prop_default_class(rel: str) -> str:
    """Start each photo on its session's class: stock YOLO drafts no fedora or cap
    box, so those photos are all hand drags and should not need a keypress first."""
    return PROP_CLASS.get(Path(rel).parts[0]) or "weapon"


def review(props: Path, photos: list[Path], max_width: int = 1280) -> None:
    win = "sherlock labels"
    cv2.namedWindow(win)
    state = {"down": None, "drag": None, "mouse": (0, 0)}
    i = 0
    while 0 <= i < len(photos):
        path = photos[i]
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        scale = min(1.0, max_width / w)
        rel = str(path.relative_to(props))
        ed = LabelEditor(read_labels(path.with_suffix(".txt"), w, h), prop_default_class(rel))

        def on_mouse(event, x, y, _flags, _param, ed=ed, scale=scale):
            state["mouse"] = (x / scale, y / scale)
            if event == cv2.EVENT_LBUTTONDOWN:
                state["down"] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and state["down"]:
                state["drag"] = (state["down"], (x, y))
            elif event == cv2.EVENT_LBUTTONUP and state["down"]:
                (x0, y0) = state["down"]
                ed.add((x0 / scale, y0 / scale), (x / scale, y / scale))
                state["down"] = state["drag"] = None
            elif event == cv2.EVENT_RBUTTONDOWN:
                ed.delete_at(x / scale, y / scale)

        cv2.setMouseCallback(win, on_mouse)
        step = 0
        while not step:
            header = (f"[{i + 1}/{len(photos)}] {rel}   class: {ed.cls + 1} {CLASSES[ed.cls]}   "
                      "drag add | right-click del | r relabel | u undo | n next | p back | q quit")
            cv2.imshow(win, _draw(img, ed, scale, state["drag"], header))
            key = cv2.waitKey(30) & 0xFF
            if ord("1") <= key < ord("1") + len(CLASSES):
                ed.cls = key - ord("1")
            elif key == ord("r"):
                ed.relabel_at(*state["mouse"])
            elif key == ord("u"):
                ed.undo()
            elif key == ord("x"):
                ed.clear()
            elif key in (ord("n"), ord(" "), 13):
                step = 1
            elif key == ord("p") and i > 0:
                step = -1
            elif key in (ord("q"), 27):
                step = len(photos)
        write_labels(path.with_suffix(".txt"), ed.labels, w, h)
        mark_reviewed(props, rel)
        i += step
    cv2.destroyAllWindows()


def status(props: Path) -> dict[str, tuple[int, int]]:
    photos = room_images(props)
    splits = room_splits(photos)
    done = load_reviewed(props)
    out = {}
    for split in ("train", "val", "test"):
        mine = [p for p in photos if splits[p] == split]
        out[split] = (sum(str(p.relative_to(props)) in done for p in mine), len(mine))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--props", type=Path, default=ROOT / "data" / "props")
    p.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    p.add_argument("--all", action="store_true", help="include photos already reviewed")
    p.add_argument("--status", action="store_true", help="print reviewed counts and exit")
    a = p.parse_args(argv)
    if a.status:
        for split, (done, total) in status(a.props).items():
            print(f"{split:5} {done}/{total} reviewed")
        return
    photos = pick(a.props, a.split, only_unreviewed=not a.all)
    if not photos:
        print("nothing to review")
        return
    review(a.props, photos)


if __name__ == "__main__":
    main()
