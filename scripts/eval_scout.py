"""Scout precision and recall, before vs after the fine-tune, on the held-out test split.

    uv run scripts/eval_scout.py --measured \\
        --after runs/scout/weights/best.pt

Before = stock YOLOv8n, its COCO names mapped onto the Scout classes (knife,
scissors, baseball bat -> weapon; bottle; person). It has no badge class, so its
badge recall is 0 by construction. After = the fine-tuned Scout. Both models see
the same images and are scored by the same code: a prediction counts only if it
has the right class and IoU >= 0.5 with an unmatched ground-truth box, at the 0.35
confidence the runtime Scout uses. In-room photos and COCO images are reported
separately: the room is where the demo runs, COCO shows nothing else broke.
Pass --smoke for pipeline checks or evaluations of smoke-trained weights; these
reports are labelled smoke and excluded from the dashboard's measured results.
Use --measured explicitly for real evaluations, including real CPU evaluations.
Smoke output defaults to eval_smoke.json, preserving the measured eval.json.
Use --baseline-only instead of --after to leave fine-tuned results unmeasured;
its default output is eval_baseline.json. --device mps selects local Apple GPU inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
from scout_data import (
    CLASSES,
    IMAGE_EXTS,
    match_counts,
    precision_recall,
    read_labels,
    scout_class,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONF = 0.35  # YoloScout's default: the operating point that matters on stage


def split_identity(dataset: Path, split: str, images: list[Path]) -> dict:
    """Identify the exact ordered class map, images and labels used by both models."""
    entries = []
    for image in sorted(images):
        label = dataset / "labels" / split / f"{image.stem}.txt"
        entries.append({
            "image": image.name,
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "label_sha256": hashlib.sha256(label.read_bytes() if label.exists() else b"").hexdigest(),
        })
    identity = {"classes": list(CLASSES), "split": split, "files": entries}
    return {"classes": list(CLASSES), "image_count": len(images),
            "dataset_fingerprint": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}


def source_of(image: Path) -> str:
    return "room" if image.name.startswith("room_") else "coco"


def ground_truth(dataset: Path, split: str) -> dict[Path, list[tuple[str, tuple]]]:
    out = {}
    for img in sorted((dataset / "images" / split).iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        h, w = cv2.imread(str(img)).shape[:2]
        labels = read_labels(dataset / "labels" / split / f"{img.stem}.txt", w, h)
        out[img] = [(CLASSES[c], box) for c, box in labels]
    return out


def predict(model_path: str, images: list[Path], conf: float = RUNTIME_CONF,
            imgsz: int = 640, device: str | None = None) -> dict[Path, list[tuple[str, float, tuple]]]:
    from ultralytics import YOLO  # lazy: torch is heavy

    model = YOLO(model_path)
    out: dict[Path, list[tuple[str, float, tuple]]] = {}
    for i in range(0, len(images), 16):
        batch = images[i:i + 16]
        for img, res in zip(batch, model.predict([str(p) for p in batch], conf=conf, imgsz=imgsz,
                                               device=device, verbose=False),
                            strict=True):
            out[img] = [(sc, float(b.conf), tuple(b.xyxy[0].tolist())) for b in res.boxes
                        if (sc := scout_class(res.names[int(b.cls)])) is not None]
    return out


def score(preds: dict[Path, list], gts: dict[Path, list], iou_thr: float = 0.5) -> dict[str, dict[str, dict]]:
    """{source: {class: {tp, fp, fn, precision, recall}}}, plus an "all" source."""
    totals: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}))
    for img, gt in gts.items():
        for cls, c in match_counts(preds.get(img, []), gt, iou_thr).items():
            for src in (source_of(img), "all"):
                for k in ("tp", "fp", "fn"):
                    totals[src][cls][k] += c[k]
    out: dict[str, dict[str, dict]] = {}
    for src, per in totals.items():
        out[src] = {}
        for cls in CLASSES:
            c = per[cls]
            p, r = precision_recall(c)
            out[src][cls] = {**c, "precision": round(p, 4), "recall": round(r, 4)}
    return out


def table(before: dict, after: dict) -> str:
    rows = [f"{'source':6} {'class':7} {'before P':>9} {'before R':>9} {'after P':>8} {'after R':>8} {'boxes':>6}"]
    for src in ("room", "coco", "all"):
        if src not in after:
            continue
        for cls in CLASSES:
            b, a = before.get(src, {}).get(cls, {}), after[src][cls]
            boxes = a["tp"] + a["fn"]
            if not boxes and not a["fp"]:
                continue
            rows.append(f"{src:6} {cls:7} {b.get('precision', 0):9.2f} {b.get('recall', 0):9.2f} "
                        f"{a['precision']:8.2f} {a['recall']:8.2f} {boxes:6d}")
    return "\n".join(rows)


def headline(before: dict, after: dict) -> str:
    src = "room" if "room" in after else "all"
    b, a = before.get(src, {}).get("weapon", {}), after[src]["weapon"]
    return (f"Scout weapon detection ({src} test photos, conf {RUNTIME_CONF}): recall {b.get('recall', 0):.0%} -> "
            f"{a['recall']:.0%}, precision {b.get('precision', 0):.0%} -> {a['precision']:.0%}")


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    evaluation = p.add_mutually_exclusive_group(required=True)
    evaluation.add_argument("--after", help="fine-tuned weights, e.g. runs/scout/weights/best.pt")
    evaluation.add_argument("--baseline-only", action="store_true", help="measure stock Scout; leave after unmeasured")
    p.add_argument("--before", default="yolov8n.pt", help="stock weights the runtime Scout uses today")
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "scout")
    p.add_argument("--split", default="test")
    p.add_argument("--conf", type=float, default=RUNTIME_CONF)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", help="inference device, e.g. mps or cpu")
    p.add_argument("--environment", default="local", help="where inference ran")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="label a pipeline/test evaluation as smoke, never dashboard evidence")
    mode.add_argument("--measured", action="store_true",
                      help="declare real training/evaluation data, eligible as measured evidence")
    p.add_argument("--out", type=Path,
                   help="default: runs/scout/eval.json, or eval_smoke.json with --smoke")
    a = p.parse_args(argv)
    if a.out is None:
        name = "eval_smoke.json" if a.smoke else "eval_baseline.json" if a.baseline_only else "eval.json"
        a.out = ROOT / "runs" / "scout" / name
    gts = ground_truth(a.dataset, a.split)
    if not gts:
        raise SystemExit(f"no images in {a.dataset / 'images' / a.split}")
    images = list(gts)
    before = score(predict(a.before, images, a.conf, a.imgsz, a.device), gts)
    after = None if a.baseline_only else score(predict(a.after, images, a.conf, a.imgsz, a.device), gts)
    sources = defaultdict(int)
    for img in images:
        sources[source_of(img)] += 1
    # Both modes run model inference; smoke scores are not measured demo evidence.
    report = {"evidence_kind": "model_evaluation",
              "execution_mode": "smoke" if a.smoke else "measured",
              "evaluation_scope": "baseline_only" if a.baseline_only else "before_after",
              "environment": a.environment, "device": a.device or "auto", "imgsz": a.imgsz,
              "split": a.split, "conf": a.conf, "iou": 0.5, "images": dict(sources),
              **split_identity(a.dataset, a.split, images),
              "before": {"model": a.before, **before},
              "after": None if after is None else {"model": a.after, **after},
              "headline": "Scout baseline only; after not measured" if after is None else headline(before, after)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(before, indent=2) if after is None else table(before, after))
    print(f"\n{report['headline']}\nwritten to {a.out}")
    return report


if __name__ == "__main__":
    main()
