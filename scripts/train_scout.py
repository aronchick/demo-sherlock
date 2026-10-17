"""Fine-tune the Scout on data/scout/scout.yaml, on whatever this machine has.

    python scripts/train_scout.py            # picks cuda > mps > cpu, with defaults sized for each
    python scripts/train_scout.py --smoke    # CPU sanity check: 1 epoch at 160 px, a minute or two
    python scripts/train_scout.py --epochs 20   # any default can be overridden

The base stays YOLOv8n so the runtime Scout keeps its ~10 ms per frame. Then:

    python scripts/eval_scout.py --after runs/scout/weights/best.pt
    YOLO_MODEL=runs/scout/weights/best.pt python run.py"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Defaults per device. cuda is sized for one H100 (80 GB): yolov8n at 640 px fits batch 64
# with room to spare and the dataset fits in RAM, so epochs are GPU-bound. mps is Apple
# silicon: unified memory, no AMP, so a smaller batch; measured on an M5 Max at ~29 s per
# epoch for 1,135 images, so 30 epochs on ~1,500 images is about 19 minutes. The local
# profiles train fewer epochs so a run fits between shooting the photos and the demo.
PROFILES = {
    "cuda": {"epochs": 60, "imgsz": 640, "batch": 64, "device": "0", "workers": 8, "cache": "ram", "patience": 15},
    "mps": {"epochs": 30, "imgsz": 640, "batch": 32, "device": "mps", "workers": 8, "cache": "ram", "patience": 10},
    "cpu": {"epochs": 30, "imgsz": 640, "batch": 16, "device": "cpu", "workers": 8, "cache": "ram", "patience": 10},
}
SMOKE = {"epochs": 1, "imgsz": 160, "batch": 4, "device": "cpu", "workers": 0, "cache": False, "patience": 0}


def pick_device() -> str:
    """Best accelerator here: an NVIDIA GPU, else Apple silicon, else the CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(data: Path, model: str = "yolov8n.pt", project: Path = ROOT / "runs", name: str = "scout",
          plots: bool = True, **overrides) -> Path:
    from ultralytics import YOLO  # lazy: importing torch is slow and optional elsewhere

    args = {**PROFILES[pick_device()], **overrides}
    print(f"training on {args['device']}: " + ", ".join(f"{k}={v}" for k, v in args.items() if k != "device"))
    m = YOLO(model)
    m.train(data=str(data), project=str(project), name=name, exist_ok=True, plots=plots, seed=0,
            cos_lr=True, close_mosaic=10, verbose=False, **args)
    best = Path(m.trainer.save_dir) / "weights" / "best.pt"
    print(f"best weights: {best}")
    return best


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "scout" / "scout.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="base weights (or a .yaml to train from scratch)")
    p.add_argument("--smoke", action="store_true", help="tiny CPU run to check the pipeline end to end")
    p.add_argument("--project", type=Path, default=ROOT / "runs")
    p.add_argument("--name", default="scout")
    for k, v in PROFILES["cuda"].items():
        p.add_argument(f"--{k}", type=type(v), default=None,
                       help="default per device: " + ", ".join(f"{d} {prof[k]}" for d, prof in PROFILES.items()))
    a = p.parse_args(argv)
    if not a.data.exists():
        raise SystemExit(f"{a.data} not found: run scripts/build_scout_dataset.py build first")
    base = SMOKE if a.smoke else {}
    given = {k: getattr(a, k) for k in PROFILES["cuda"] if getattr(a, k) is not None}
    return train(a.data, a.model, a.project, a.name, plots=not a.smoke, **{**base, **given})


if __name__ == "__main__":
    main()
