"""Measure Scout confidence tradeoffs on the exact saved baseline dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

from eval_scout import ground_truth, predict, score, split_identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--conf", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.25, 0.35])
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--measured", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.resolve() == args.baseline.resolve():
        parser.error("output must be a new file; never overwrite the baseline")
    if not args.conf or any(not 0 < value < 1 for value in args.conf):
        parser.error("confidence values must be between 0 and 1")
    baseline_bytes = args.baseline.read_bytes()
    baseline = json.loads(baseline_bytes)
    gts = ground_truth(args.dataset, baseline["split"])
    images = list(gts)
    identity = split_identity(args.dataset, baseline["split"], images)
    if not images or any(baseline.get(key) != value for key, value in identity.items()):
        parser.error("dataset or ordered class map differs from the saved baseline")
    weights_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if weights_hash != baseline["provenance"]["weights_sha256"]:
        parser.error("weights differ from the saved baseline")
    if args.measured and (baseline.get("evidence_kind") != "model_evaluation"
                          or baseline.get("execution_mode") != "measured"):
        parser.error("measured sweep requires a measured model-evaluation baseline")
    curve = []
    for conf in sorted(set(args.conf)):
        predictions = predict(str(args.weights), images, conf, baseline["imgsz"], args.device)
        metrics = score(predictions, gts, baseline["iou"])
        curve.append({"conf": conf, "metrics": metrics})
        weapon = metrics["all"]["weapon"]
        print(f"conf={conf:.2f}: precision={weapon['precision']:.4f} recall={weapon['recall']:.4f} "
              f"TP={weapon['tp']} FP={weapon['fp']} FN={weapon['fn']}", flush=True)
    report = {
        "evidence_kind": "model_evaluation", "execution_mode": "measured" if args.measured else "smoke",
        "evaluation_scope": "confidence_sweep", "environment": "local", "device": args.device,
        "created_at": datetime.now(UTC).isoformat(), **identity, "split": baseline["split"],
        "images": baseline["images"], "model": baseline["before"]["model"],
        "weights_sha256": weights_hash, "imgsz": baseline["imgsz"], "iou": baseline["iou"],
        "baseline_file": args.baseline.name,
        "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "ultralytics")},
        "curve": curve,
        "limitations": [
            "Same standalone licensed COCO val subset as baseline; not the fine-tune held-out room set.",
            "Per-image box metrics, not temporal live-event recall or measured downstream Verifier load.",
            "Weapon means knife, scissors and baseball bat; no firearm evaluation.",
            "No badge/fedora/cap ground truth in this subset; zero-support rows do not measure recall.",
        ],
    }
    if args.baseline.read_bytes() != baseline_bytes:
        raise RuntimeError("baseline changed during inference")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(f"Written to {args.out}")
    return report


if __name__ == "__main__":
    main()
