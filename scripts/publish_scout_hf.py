"""Stage the fine-tuned Scout and its open dataset for Hugging Face. Uploads ONLY
with --publish, which needs the owner's go-ahead and HF_TOKEN in the environment.

    python scripts/publish_scout_hf.py --model-repo ORG/sherlock-scout \\
        --dataset-repo ORG/sherlock-scout-data           # dry run: stage + list
    python scripts/publish_scout_hf.py ... --publish      # upload (private repos)

Stages runs/hf_staging/{model,dataset} with cards, then lists every file.
Repos are created private unless --public.

Licences. The dataset is CC-BY-4.0: COCO annotations are CC-BY-4.0, and the
builder keeps only images whose Flickr licence allows CC-BY redistribution
(per-image licence and URL in manifest.jsonl). The weights are fine-tuned from
Ultralytics YOLOv8n, which is AGPL-3.0, so the model card defaults to AGPL-3.0;
--model-license overrides it if the owner decides otherwise.

Consent. In-room photos show real people, so they stay out of the dataset upload
unless --include-room-photos, which needs everyone pictured to agree."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from scout_data import CC_BY_LICENSES, CLASSES
from sherlock.dashboard.numbers import count, number, scout_result

ROOT = Path(__file__).resolve().parents[1]


def measured_headline(ev) -> str:
    """Use the dashboard's provenance/row checks and require complete score fields."""
    error = "Incomplete, smoke or unlabelled Scout evaluation cannot be published"
    if not isinstance(ev, dict):
        raise TypeError(error)
    try:
        result = scout_result(ev)
    except (TypeError, KeyError):
        result = None
    images = ev.get("images")
    image_counts = [count(n) for n in images.values()] if isinstance(images, dict) else []
    if (
        result is None
        or any(result[key] is None for key in ("before_precision", "after_precision", "before_recall", "after_recall"))
        or number(ev.get("conf"), maximum=1) is None
        or number(ev.get("iou"), minimum=0.01, maximum=1) is None
        or not image_counts or None in image_counts or not sum(image_counts)
    ):
        raise ValueError(error)
    for side in ("before", "after"):
        row = ev[side][result["source"]]["weapon"]
        if any(count(row.get(key)) is None for key in ("tp", "fp", "fn")):
            raise ValueError(error)
    return (f"Scout weapon detection ({result['source']} {result['split']} photos, conf {ev['conf']:g}): "
            f"recall {result['before_recall']:.0%} -> {result['after_recall']:.0%}, "
            f"precision {result['before_precision']:.0%} -> {result['after_precision']:.0%}")


def model_card(repo: str, dataset_repo: str, license_id: str, headline: str | None) -> str:
    headline = headline if headline is not None else "Run scripts/eval_scout.py --measured to fill this in."
    return f"""---
license: {license_id}
library_name: ultralytics
pipeline_tag: object-detection
tags: [yolov8, object-detection, security, sherlock]
datasets: [{dataset_repo}]
---
# {repo.split('/')[-1]}

The Scout of [Sherlock](https://github.com/aronchick/demo-sherlock): a YOLOv8n detector
fine-tuned to flag `{"`, `".join(CLASSES)}` so a security camera only calls its vision-language
models when something worth a closer look appears.

**Result on held-out test photos:** {headline}

## Use
```python
from ultralytics import YOLO
YOLO("best.pt").predict("frame.jpg", conf=0.35)
```
In Sherlock: `YOLO_MODEL=best.pt python run.py`.

## Training data
An open COCO 2017 subset (knife, scissors, baseball bat -> `weapon`; bottle; person) limited
to CC-BY-compatible images, plus photos of props shot in the demo room. See
[{dataset_repo}](https://huggingface.co/datasets/{dataset_repo}).

## Limits
- No firearms: COCO has none and we did not fake any.
- `badge` comes only from a small set of in-room photos; expect it to be the weakest class.
- A detector, not an identifier: it says where a person is, never who.

## Licence
Base model Ultralytics YOLOv8n (AGPL-3.0). This card: `{license_id}`.
"""


def dataset_card(repo: str, n: dict, rooms: bool) -> str:
    room_line = ("Includes in-room photos of the demo props, published with the consent of everyone pictured."
                 if rooms else "In-room photos are not included: they show real people.")
    return f"""---
license: cc-by-4.0
task_categories: [object-detection]
tags: [yolo, coco, security, sherlock]
---
# {repo.split('/')[-1]}

YOLO-format detection data for the [Sherlock](https://github.com/aronchick/demo-sherlock) Scout.
Classes: {", ".join(f"{i} `{c}`" for i, c in enumerate(CLASSES))}. Splits: {json.dumps(n)}.

## Sources and attribution
- **COCO 2017** (Lin et al., 2014). Annotations: COCO Consortium, CC-BY-4.0. Images: Flickr,
  restricted to licence ids {sorted(CC_BY_LICENSES)} (Attribution; No known copyright
  restrictions; United States Government Work). Each image's licence and source URL is in
  `manifest.jsonl`; please keep that attribution when you reuse an image.
- {room_line}

Labels for COCO images are COCO's own boxes mapped onto the Scout classes (crowd boxes dropped).
"""


def stage(weights: Path, dataset: Path, out: Path, model_repo: str, dataset_repo: str,
          license_id: str, include_rooms: bool, eval_json: Path | None, coco_manifest: Path) -> dict[str, Path]:
    headline = None
    if eval_json and eval_json.exists():
        headline = measured_headline(json.loads(eval_json.read_text()))
    if out.exists():
        shutil.rmtree(out)
    mdir, ddir = out / "model", out / "dataset"
    mdir.mkdir(parents=True)
    ddir.mkdir(parents=True)

    shutil.copyfile(weights, mdir / "best.pt")
    if headline is not None:
        shutil.copyfile(eval_json, mdir / "eval.json")
    (mdir / "README.md").write_text(model_card(model_repo, dataset_repo, license_id, headline))

    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        imgs = sorted((dataset / "images" / split).glob("*"))
        keep = [p for p in imgs if p.name.startswith("coco_") or (include_rooms and p.name.startswith("room_"))]
        (ddir / "images" / split).mkdir(parents=True)
        (ddir / "labels" / split).mkdir(parents=True)
        for img in keep:
            shutil.copyfile(img, ddir / "images" / split / img.name)
            lbl = dataset / "labels" / split / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copyfile(lbl, ddir / "labels" / split / lbl.name)
        counts[split] = len(keep)
    if coco_manifest.exists():
        shutil.copyfile(coco_manifest, ddir / "manifest.jsonl")
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (ddir / "scout.yaml").write_text(f"path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                     f"names:\n{names}\n")
    (ddir / "README.md").write_text(dataset_card(dataset_repo, counts, include_rooms))
    return {"model": mdir, "dataset": ddir}


def upload(staged: dict[str, Path], model_repo: str, dataset_repo: str, public: bool) -> None:
    from huggingface_hub import HfApi  # only on --publish

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (load it from Doppler); nothing uploaded")
    api = HfApi(token=token)
    for kind, repo in (("model", model_repo), ("dataset", dataset_repo)):
        api.create_repo(repo, repo_type=kind, private=not public, exist_ok=True)
        api.upload_folder(folder_path=str(staged[kind]), repo_id=repo, repo_type=kind,
                          commit_message="Sherlock Scout fine-tune")
        print(f"uploaded https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}")


def main(argv: list[str] | None = None) -> dict[str, Path]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-repo", required=True)
    p.add_argument("--dataset-repo", required=True)
    p.add_argument("--weights", type=Path, default=ROOT / "runs" / "scout" / "weights" / "best.pt")
    p.add_argument("--eval", type=Path, default=ROOT / "runs" / "scout" / "eval.json")
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "scout")
    p.add_argument("--coco-manifest", type=Path, default=ROOT / "data" / "coco_subset" / "manifest.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "hf_staging")
    p.add_argument("--model-license", default="agpl-3.0", help="see the licence note above")
    p.add_argument("--include-room-photos", action="store_true", help="only with consent of everyone pictured")
    p.add_argument("--public", action="store_true", help="default: private repos")
    p.add_argument("--publish", action="store_true", help="actually upload (owner approval required)")
    a = p.parse_args(argv)
    if not a.weights.exists():
        raise SystemExit(f"{a.weights} not found: train first")
    staged = stage(a.weights, a.dataset, a.out, a.model_repo, a.dataset_repo, a.model_license,
                   a.include_room_photos, a.eval, a.coco_manifest)
    for kind, d in staged.items():
        files = [f for f in d.rglob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in files) / 1e6
        print(f"{kind:7} {len(files):5d} files {size:8.1f} MB  {d}")
    if a.publish:
        upload(staged, a.model_repo, a.dataset_repo, a.public)
    else:
        print("dry run: nothing uploaded. Re-run with --publish once the owner approves.")
    return staged


if __name__ == "__main__":
    main()
