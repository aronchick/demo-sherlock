"""Draft the in-room Scout labels with Gemini 3.8 Flash through the Antigravity CLI
(agy), which sees the props stock YOLO cannot (fedora, cap, coke) and boxes the rest.

    uv run scripts/gemini_prelabel.py                     # every photo without a Gemini draft
    uv run scripts/gemini_prelabel.py --jobs 6 --batch 10

Each batch of downscaled photos goes to one `agy -p` call with a JSON schema; boxes come
back as [ymin, xmin, ymax, xmax] on 0-1000 and replace the photo's YOLO draft. Raw
answers are cached in data/gemini_labels/, so a rerun sends only what is missing.

Drafts stay drafts: review_labels.py marks photos reviewed, and build refuses test
photos that are not, so the eval is graded on labels a person confirmed. The photos
show real people and this sends them to Google: run it only with the owner's go-ahead."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scout_data import (
    CLASSES,
    PROP_CLASS,
    load_reviewed,
    prop_of,
    room_images,
    write_labels,
)

MODEL = "gemini-3.8-flash-medium"
LABELS = ("person", "weapon", "fedora", "cap", "coke")  # what Gemini is asked for; all are CLASSES
SEND_WIDTH = 1280  # boxes come back normalised, so a smaller upload loses nothing
SCHEMA = {
    "type": "object",
    "properties": {"images": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "file": {"type": "string"},
            "boxes": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": list(LABELS)},
                    "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                },
                "required": ["label", "box_2d"],
            }},
        },
        "required": ["file", "boxes"],
    }}},
    "required": ["images"],
}
PROMPT = (
    "Open and look at every image file in this directory ({files}), one at a time. For each, "
    "return a bounding box for every visible instance of: person (the whole visible body), "
    "weapon (a knife, scissors or bat), fedora, cap (a baseball cap, worn or held), coke (a soda "
    "can). box_2d is [ymin, xmin, ymax, xmax] normalized to 0-1000. Include only objects that are "
    "actually visible; an image may have none of the props. Return one entry per file."
)


def yolo_labels(boxes: list[dict]) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Gemini boxes -> (class id, x1 y1 x2 y2 on a 1000x1000 canvas); malformed boxes dropped."""
    out = []
    for b in boxes:
        y1, x1, y2, x2 = b["box_2d"]
        if b["label"] in LABELS and 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000:
            out.append((CLASSES.index(b["label"]), (x1, y1, x2, y2)))
    return out


def ask_gemini(photos: list[Path], model: str, timeout: float) -> dict[Path, list[dict]]:
    """One agy call for a batch. Photos missing from the answer are left out, not guessed."""
    with tempfile.TemporaryDirectory(prefix="gemini-label-") as tmp:
        names = {}
        for i, p in enumerate(photos):
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            if w > SEND_WIDTH:
                img = cv2.resize(img, (SEND_WIDTH, round(h * SEND_WIDTH / w)))
            name = f"img{i:02d}.jpg"
            cv2.imwrite(str(Path(tmp) / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            names[name] = p
        schema = Path(tmp) / "schema.json"
        schema.write_text(json.dumps(SCHEMA))
        cmd = ["agy", "-p", PROMPT.format(files=", ".join(names)), "--model", model, "--add-dir", tmp,
               "--json-schema", str(schema), "--output-format", "json", "--print-timeout", f"{timeout:.0f}s"]
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout + 60, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"agy exited {r.returncode}: {r.stderr.strip()[-300:]}")
    answer = json.loads(r.stdout)
    if answer.get("status") != "SUCCESS":
        raise RuntimeError(f"agy status {answer.get('status')}")
    images = (answer.get("structured_output") or {}).get("images", [])
    return {names[im["file"]]: im["boxes"] for im in images if im.get("file") in names}


def main(argv: list[str] | None = None) -> dict[Path, list[dict]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--props", type=Path, default=ROOT / "data" / "props")
    p.add_argument("--cache", type=Path, help="raw answers (default: <props>/../gemini_labels)")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--batch", type=int, default=10, help="photos per agy call")
    p.add_argument("--jobs", type=int, default=4, help="agy calls at once")
    p.add_argument("--timeout", type=float, default=900, help="seconds per agy call")
    p.add_argument("--limit", type=int, help="label at most this many photos (a trial run)")
    a = p.parse_args(argv)
    cache = a.cache or a.props.parent / "gemini_labels"
    reviewed = load_reviewed(a.props)

    def cached(photo: Path) -> Path:
        return (cache / photo.relative_to(a.props)).with_suffix(".json")

    todo = [ph for ph in room_images(a.props)
            if str(ph.relative_to(a.props)) not in reviewed and not cached(ph).exists()][:a.limit]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    print(f"{len(todo)} photos in {len(batches)} batches, {a.jobs} at a time, {a.model}")
    done: dict[Path, list[dict]] = {}
    t0 = time.monotonic()
    with ThreadPoolExecutor(a.jobs) as pool:
        futures = {pool.submit(ask_gemini, b, a.model, a.timeout): b for b in batches}
        for f in as_completed(futures):
            batch = futures[f]
            try:
                got = f.result()
            except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as e:
                print(f"batch of {len(batch)} from {batch[0].relative_to(a.props)} failed: {e}")
                continue
            for photo, boxes in got.items():
                c = cached(photo)
                c.parent.mkdir(parents=True, exist_ok=True)
                c.write_text(json.dumps({"model": a.model, "boxes": boxes}))
                write_labels(photo.with_suffix(".txt"), yolo_labels(boxes), 1000, 1000)
                done[photo] = boxes
            print(f"{len(done)}/{len(todo)} labelled, {time.monotonic() - t0:.0f} s", flush=True)
    summarise(a.props, cache)
    return done


def summarise(props: Path, cache: Path) -> None:
    """Per prop: how many photos Gemini found the session's prop in. The misses are the
    photos to look at first in review."""
    by_prop: dict[str, list[Path]] = {}
    for photo in room_images(props):
        by_prop.setdefault(prop_of(photo), []).append(photo)
    for prop, photos in sorted(by_prop.items()):
        want = PROP_CLASS.get(prop)
        answers = [json.loads(c.read_text())["boxes"] for ph in photos
                   if (c := (cache / ph.relative_to(props)).with_suffix(".json")).exists()]
        hit = sum(any(b["label"] == want for b in boxes) for boxes in answers)
        print(f"{prop:8s} {len(answers)}/{len(photos)} labelled" + (f", prop found in {hit}" if want else ""))


if __name__ == "__main__":
    main()
