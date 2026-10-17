"""Build the Scout fine-tune dataset: an open COCO subset plus in-room photos,
mapped onto the Scout classes (person, weapon, badge, bottle).

    python scripts/build_scout_dataset.py coco              # open data, CC-BY-compatible images only
    python scripts/build_scout_dataset.py prelabel          # stock YOLOv8 drafts labels for data/props
    python scripts/review_labels.py --split test            # fix labels by hand (test split first)
    python scripts/build_scout_dataset.py build             # -> data/scout/scout.yaml

coco: reads COCO 2017 instance annotations, keeps images that contain a knife,
scissors, baseball bat or bottle under a licence that allows CC-BY redistribution,
caps each category, downloads the images and writes YOLO labels for every person,
weapon and bottle in them. A manifest records each image's licence and source URL
for attribution.

prelabel: runs stock YOLOv8 over data/props and writes draft labels next to each
photo. Drafts are a starting point, not ground truth: review them, at least the
test split, before trusting the before/after numbers.

build: merges both into data/scout/{images,labels}/{train,val,test}. COCO images
split by a stable hash of the image id; in-room photos split by time within each
session so near-duplicate neighbours never straddle train and test."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scout_data import (
    CC_BY_LICENSES,
    CLASSES,
    COCO_TARGETS,
    COCO_TO_SCOUT,
    PROP_HINTS,
    coco_xyxy,
    hash_split,
    load_reviewed,
    prop_of,
    room_images,
    room_splits,
    scout_class,
    write_labels,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DEFAULT_CAPS = {"knife": 600, "scissors": 400, "baseball bat": 600, "bottle": 400}


# --- coco -----------------------------------------------------------------------------
def fetch_annotations(cache: Path) -> list[Path]:
    """Download (241 MB, once) and unpack the two instance files we need."""
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / "annotations_trainval2017.zip"
    if not zpath.exists():
        print(f"downloading {COCO_ANN_URL}")
        urllib.request.urlretrieve(COCO_ANN_URL, zpath)
    out = []
    with zipfile.ZipFile(zpath) as z:
        for split in ("train2017", "val2017"):
            name = f"annotations/instances_{split}.json"
            dest = cache / Path(name).name
            if not dest.exists():
                z.extract(name, cache)
                (cache / name).rename(dest)
            out.append(dest)
    return out


def select_coco(ann: dict, caps: dict[str, int], licenses: frozenset[int] = CC_BY_LICENSES) -> dict:
    """Pick images for the subset. Returns {image_id: {"image": info, "boxes": [(cls, xyxy)]}}.

    Only images under an allowed licence; at most caps[cat] images per target
    category (an image counts toward every target it contains); crowd boxes skipped."""
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    images = {im["id"]: im for im in ann["images"] if im.get("license") in licenses}
    boxes: dict[int, list] = defaultdict(list)
    has: dict[int, set[str]] = defaultdict(set)
    for a in ann["annotations"]:
        name = cats.get(a["category_id"], "")
        if a["image_id"] not in images or a.get("iscrowd") or name not in COCO_TO_SCOUT:
            continue
        boxes[a["image_id"]].append((CLASSES.index(COCO_TO_SCOUT[name]), coco_xyxy(a["bbox"])))
        if name in COCO_TARGETS:
            has[a["image_id"]].add(name)
    taken: Counter[str] = Counter()
    picked: dict[int, dict] = {}
    # Rarest category first, so bottles (common) can't crowd out scissors (rare).
    for cat in sorted(caps, key=lambda c: sum(c in h for h in has.values())):
        for iid in sorted(i for i, h in has.items() if cat in h):
            if taken[cat] >= caps[cat]:
                break
            if iid not in picked:
                picked[iid] = {"image": images[iid], "boxes": boxes[iid]}
                taken.update(has[iid])
    return picked


def _download(item: tuple[str, Path]) -> bool:
    url, dest = item
    if dest.exists():
        return True
    try:
        urllib.request.urlretrieve(url, dest)
    except OSError as e:
        print(f"skip {url}: {e}")
        return False
    return True


def cmd_coco(a: argparse.Namespace) -> None:
    files = [Path(f) for f in a.annotations] if a.annotations else fetch_annotations(DATA / "coco_cache")
    caps = {**DEFAULT_CAPS, **dict(a.cap or [])}
    out = a.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    manifest, jobs = [], []
    lic_names: dict[int, str] = {}
    for f in files:
        ann = json.loads(f.read_text())
        lic_names.update({lic["id"]: lic["name"] for lic in ann.get("licenses", [])})
        for iid, rec in select_coco(ann, caps).items():
            im = rec["image"]
            stem = f"coco_{iid:012d}"
            jobs.append((im["coco_url"], out / "images" / f"{stem}.jpg"))
            write_labels(out / "labels" / f"{stem}.txt", rec["boxes"], im["width"], im["height"])
            manifest.append({"file": f"{stem}.jpg", "coco_id": iid, "license": im["license"],
                             "license_name": lic_names.get(im["license"], ""),
                             "coco_url": im["coco_url"], "flickr_url": im.get("flickr_url", "")})
    print(f"downloading {len(jobs)} images ({a.workers} at a time)")
    with ThreadPoolExecutor(a.workers) as pool:
        ok = list(pool.map(_download, jobs))
    manifest = [m for m, good in zip(manifest, ok, strict=True) if good]
    for (_, dest), good in zip(jobs, ok, strict=True):
        if not good:
            (out / "labels" / f"{dest.stem}.txt").unlink(missing_ok=True)  # no image, no label
    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    per = Counter(lic_names.get(m["license"], m["license"]) for m in manifest)
    print(f"{len(manifest)} COCO images in {out} (licences: {dict(per)})")


# --- prelabel -------------------------------------------------------------------------
def draft_labels(names: dict[int, str], dets: list[tuple[int, float, tuple]], prop: str) -> list[tuple[int, tuple]]:
    """Stock-YOLO detections -> Scout labels for one in-room photo. Weapon-ish COCO
    names become weapon; in a badge session, things stock YOLO mistakes a badge for
    (phone, remote, book) become badge drafts for the reviewer to keep or delete."""
    out = []
    for cls_id, _, box in dets:
        name = names[cls_id]
        if name in PROP_HINTS.get(prop, ()):
            out.append((CLASSES.index("badge"), box))
        elif (sc := scout_class(name)) is not None:
            out.append((CLASSES.index(sc), box))
    return out


def cmd_prelabel(a: argparse.Namespace) -> None:
    from ultralytics import YOLO  # lazy: the rest of the kit runs without torch

    model = YOLO(a.model)
    images = room_images(a.props)
    reviewed = load_reviewed(a.props)
    todo = [p for p in images if a.overwrite or not p.with_suffix(".txt").exists()]
    todo = [p for p in todo if str(p.relative_to(a.props)) not in reviewed]
    print(f"pre-labelling {len(todo)} of {len(images)} photos with {a.model}")
    for i in range(0, len(todo), 16):
        batch = todo[i:i + 16]
        for path, res in zip(batch, model.predict([str(p) for p in batch], conf=a.conf, verbose=False),
                             strict=True):
            dets = [(int(b.cls), float(b.conf), tuple(b.xyxy[0].tolist())) for b in res.boxes]
            h, w = res.orig_shape
            write_labels(path.with_suffix(".txt"), draft_labels(res.names, dets, prop_of(path)), w, h)
    print("drafts written; next: python scripts/review_labels.py --split test   (then the rest)")


# --- build ----------------------------------------------------------------------------
def _place(src_img: Path, src_lbl: Path | None, out: Path, split: str, name: str) -> None:
    shutil.copyfile(src_img, out / "images" / split / name)
    dst = out / "labels" / split / f"{Path(name).stem}.txt"
    if src_lbl is not None and src_lbl.exists():
        shutil.copyfile(src_lbl, dst)
    else:
        dst.write_text("")  # a background image: no objects, still a useful negative


def cmd_build(a: argparse.Namespace) -> None:
    out = a.out
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    counts: Counter[tuple[str, str]] = Counter()

    coco = a.coco / "images"
    for img in sorted(coco.glob("*.jpg")) if coco.exists() else []:
        split = hash_split(img.stem)
        _place(img, a.coco / "labels" / f"{img.stem}.txt", out, split, img.name)
        counts["coco", split] += 1

    photos = room_images(a.props)
    splits = room_splits(photos)
    reviewed = load_reviewed(a.props)
    unreviewed_test = [p for p in photos if splits[p] == "test" and str(p.relative_to(a.props)) not in reviewed]
    if unreviewed_test and not a.allow_unreviewed_test:
        raise SystemExit(f"{len(unreviewed_test)} in-room test photos still carry unreviewed stock-YOLO drafts; "
                         "the before/after eval would grade the stock model against its own guesses.\n"
                         "Run: python scripts/review_labels.py --split test   (or pass --allow-unreviewed-test)")
    for p in photos:
        rel = p.relative_to(a.props)
        name = f"room_{rel.parts[0]}_{rel.parts[1]}_{p.stem}{p.suffix.lower()}"
        _place(p, p.with_suffix(".txt"), out, splits[p], name)
        counts["room", splits[p]] += 1

    yaml = out / "scout.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    yaml.write_text(f"# Sherlock Scout fine-tune. Built by scripts/build_scout_dataset.py\n"
                    f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    f"names:\n{names}\n")
    print(f"{yaml}\n" + "\n".join(f"  {src:5} {split:5} {n}" for (src, split), n in sorted(counts.items())))
    if not counts:
        print("empty: run `coco` and/or capture_props.py first")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("coco", help="open COCO subset (CC-BY-compatible licences)")
    c.add_argument("--annotations", nargs="*", help="instances_*.json files (default: download)")
    c.add_argument("--cap", nargs=2, action="append", metavar=("CATEGORY", "N"),
                   type=lambda s: int(s) if s.isdigit() else s, help="max images per category")
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--out", type=Path, default=DATA / "coco_subset")
    c.set_defaults(fn=cmd_coco)

    pl = sub.add_parser("prelabel", help="draft labels for data/props with stock YOLOv8")
    pl.add_argument("--props", type=Path, default=DATA / "props")
    pl.add_argument("--model", default="yolov8n.pt")
    pl.add_argument("--conf", type=float, default=0.25)
    pl.add_argument("--overwrite", action="store_true", help="redo drafts (never touches reviewed photos)")
    pl.set_defaults(fn=cmd_prelabel)

    b = sub.add_parser("build", help="merge into data/scout and write scout.yaml")
    b.add_argument("--coco", type=Path, default=DATA / "coco_subset")
    b.add_argument("--props", type=Path, default=DATA / "props")
    b.add_argument("--out", type=Path, default=DATA / "scout")
    b.add_argument("--allow-unreviewed-test", action="store_true")
    b.set_defaults(fn=cmd_build)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
