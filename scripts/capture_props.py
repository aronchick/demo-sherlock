"""Shoot in-room training photos of one prop from a named camera.

    uv run scripts/capture_props.py --camera hero --prop knife --n 40 --interval 0.7

Grabs the camera's wide view (what the Scout sees at runtime) every --interval
seconds into data/props/<prop>/<session>/NNNN.jpg. One session per prop per pass;
move the prop around, change hands, distance and angle while it runs. Aim for
100-200 frames across knife, bat, scissors, badge and bottle, plus one "none"
session (empty room, a person with nothing in hand) for hard negatives.

The camera comes from CAMERAS in .env (same names as the dashboard), or --source
overrides it (a device index like 0, a file, or "synthetic" for a dry run)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sherlock"))

from scout_data import PROP_CLASS
from sherlock.camera import build_camera
from sherlock.config import Config


def open_camera(name: str, source: str | None):
    if source is not None:
        return build_camera(name, "fake", source)
    for c in Config().cameras:
        if c.name == name:
            return build_camera(c.name, c.backend, c.source)
    known = [c.name for c in Config().cameras]
    raise SystemExit(f"no camera named {name!r} in CAMERAS (known: {known}); pass --source")


def capture(cam, out_dir: Path, n: int, interval: float, countdown: float = 0.0) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if countdown:
        print(f"starting in {countdown:.0f} s: get the prop in frame")
        time.sleep(countdown)
    saved: list[Path] = []
    next_at = time.monotonic()
    while len(saved) < n:
        frame = cam.wide_frame()
        path = out_dir / f"{len(saved):04d}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise SystemExit(f"could not write {path}")
        saved.append(path)
        print(f"\r{len(saved)}/{n} {path}", end="", flush=True)
        next_at += interval
        time.sleep(max(0.0, next_at - time.monotonic()))
    print()
    return saved


def main(argv: list[str] | None = None) -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="hero", help="camera name from CAMERAS")
    p.add_argument("--source", help="override: device index, file or 'synthetic'")
    p.add_argument("--prop", required=True, choices=sorted(PROP_CLASS))
    p.add_argument("--n", type=int, default=40, help="frames to grab")
    p.add_argument("--interval", type=float, default=0.7, help="seconds between frames")
    p.add_argument("--countdown", type=float, default=3.0)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "props")
    a = p.parse_args(argv)
    session = f"{a.camera}-{time.strftime('%Y%m%d-%H%M%S')}"
    saved = capture(open_camera(a.camera, a.source), a.out / a.prop / session, a.n, a.interval, a.countdown)
    print(f"{len(saved)} frames of {a.prop!r} in {a.out / a.prop / session}")
    print("next: uv run --extra yolo scripts/build_scout_dataset.py prelabel")
    return saved


if __name__ == "__main__":
    main()
