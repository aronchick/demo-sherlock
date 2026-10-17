"""Scout fine-tune kit: class map, labels, COCO selection, splits, metrics, the
label editor, dataset build, capture, a dry-run publish, and a tiny CPU train+eval."""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_scout_dataset as bsd
import capture_props
import eval_scout
import publish_scout_hf
import review_labels
import scout_data as sd
import train_scout


# --- vocabulary and labels ------------------------------------------------------------
def test_class_map_is_consistent():
    assert set(sd.COCO_TO_SCOUT.values()) <= set(sd.CLASSES)
    assert {c for c in sd.PROP_CLASS.values() if c} <= set(sd.CLASSES)
    assert sd.scout_class("knife") == sd.scout_class("baseball bat") == sd.scout_class("weapon") == "weapon"
    assert sd.scout_class("person") == "person" and sd.scout_class("cup") is None
    assert sd.CLASSES[0] == "person", "person stays a class so weapon events keep a head to zoom on"


def test_yolo_labels_round_trip_and_clip(tmp_path):
    f = tmp_path / "a.txt"
    sd.write_labels(f, [(1, (10, 20, 110, 220)), (3, (-5, 0, 50, 40)), (0, (5, 5, 6, 6))], 200, 300)
    back = sd.read_labels(f, 200, 300)
    assert len(back) == 2, "boxes under 2 px are dropped"
    assert back[0][0] == 1 and np.allclose(back[0][1], (10, 20, 110, 220), atol=0.01)
    assert np.allclose(back[1][1], (0, 0, 50, 40), atol=0.01), "clipped to the image"
    assert sd.coco_xyxy([10, 20, 30, 40]) == (10, 20, 40, 60)


# --- COCO selection -------------------------------------------------------------------
def _ann():
    cats = [{"id": 1, "name": "person"}, {"id": 44, "name": "bottle"}, {"id": 49, "name": "knife"},
            {"id": 87, "name": "scissors"}, {"id": 62, "name": "chair"}]
    images = [{"id": i, "license": lic, "width": 100, "height": 100, "coco_url": f"http://x/{i}.jpg"}
              for i, lic in [(1, 4), (2, 4), (3, 1), (4, 7), (5, 8), (6, 4)]]
    a = [(1, 49, 0), (1, 1, 0), (2, 49, 0), (3, 49, 0), (4, 87, 0), (4, 62, 0), (5, 44, 1), (6, 44, 0)]
    anns = [{"image_id": i, "category_id": c, "iscrowd": crowd, "bbox": [1, 2, 30, 40]} for i, c, crowd in a]
    return {"categories": cats, "images": images, "annotations": anns}


def test_select_coco_keeps_open_licences_skips_crowds_and_caps():
    picked = bsd.select_coco(_ann(), {"knife": 1, "scissors": 5, "bottle": 5, "baseball bat": 5})
    assert 3 not in picked, "licence 1 is NonCommercial"
    assert 5 not in picked, "its only target box is a crowd box"
    assert sorted(picked) == [1, 4, 6], "knife capped at one image; scissors and bottle kept"
    assert sorted(c for c, _ in picked[1]["boxes"]) == [sd.CLASSES.index("person"), sd.CLASSES.index("weapon")]
    assert [c for c, _ in picked[4]["boxes"]] == [sd.CLASSES.index("weapon")], "chair is not a Scout class"
    taken = bsd.Counter({"knife": 1})  # a knife image already came from the other annotation file
    assert 1 not in bsd.select_coco(_ann(), {"knife": 1}, taken=taken), "caps span train2017 and val2017"


# --- splits and metrics ---------------------------------------------------------------
def test_room_splits_hold_out_the_end_of_each_session():
    paths = [Path(f"props/{prop}/s1/{i:04d}.jpg") for prop in ("knife", "badge") for i in range(10)]
    splits = sd.room_splits(reversed(paths))
    for prop in ("knife", "badge"):
        got = [splits[Path(f"props/{prop}/s1/{i:04d}.jpg")] for i in range(10)]
        assert got == ["train"] * 7 + ["val"] + ["test"] * 2
    assert sd.hash_split("coco_000000000123") == sd.hash_split("coco_000000000123")


def test_match_counts_precision_recall():
    gts = [("weapon", (0, 0, 10, 10)), ("weapon", (50, 50, 60, 60)), ("person", (0, 0, 100, 100))]
    preds = [("weapon", 0.9, (0, 0, 10, 11)), ("weapon", 0.8, (0, 0, 10, 10)),  # 2nd is a duplicate
             ("bottle", 0.7, (50, 50, 60, 60)), ("person", 0.6, (0, 0, 100, 100))]
    c = sd.match_counts(preds, gts)
    assert c["weapon"] == {"tp": 1, "fp": 1, "fn": 1}
    assert c["bottle"] == {"tp": 0, "fp": 1, "fn": 0} and c["person"]["tp"] == 1
    assert sd.precision_recall(c["weapon"]) == (0.5, 0.5)


# --- labelling ------------------------------------------------------------------------
def test_draft_labels_map_stock_names_and_badge_hints():
    names = {0: "person", 43: "knife", 67: "cell phone", 41: "cup"}
    dets = [(0, 0.9, (0, 0, 9, 9)), (43, 0.8, (1, 1, 5, 5)), (67, 0.5, (2, 2, 6, 6)), (41, 0.5, (3, 3, 7, 7))]
    as_cls = [sd.CLASSES[c] for c, _ in bsd.draft_labels(names, dets, "badge")]
    assert as_cls == ["person", "weapon", "badge"], "cup dropped; the phone is a badge draft in a badge session"
    assert [sd.CLASSES[c] for c, _ in bsd.draft_labels(names, dets, "knife")] == ["person", "weapon"]


def test_label_editor_add_delete_relabel_undo():
    ed = review_labels.LabelEditor([(0, (0, 0, 100, 100))])
    assert not ed.add((5, 5), (6, 6)), "a mis-click is not a box"
    assert ed.add((40, 40), (10, 10)) and ed.labels[-1] == (1, (10, 10, 40, 40))
    ed.cls = 2
    assert ed.relabel_at(20, 20) and ed.labels[1][0] == 2, "the smallest box under the pointer"
    assert ed.delete_at(20, 20) and ed.labels == [(0, (0, 0, 100, 100))]
    ed.undo()
    assert len(ed.labels) == 2
    ed.clear()
    ed.undo()
    assert len(ed.labels) == 2


# --- build, capture, publish ----------------------------------------------------------
def _img(path: Path, color=(40, 40, 40), w=96, h=96, box=None):
    im = np.full((h, w, 3), color, np.uint8)
    if box:
        cv2.rectangle(im, box[:2], box[2:], (230, 230, 230), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), im)


def _fake_sources(tmp_path) -> tuple[Path, Path]:
    coco = tmp_path / "coco_subset"
    for i in range(6):
        stem = f"coco_{i:012d}"
        _img(coco / "images" / f"{stem}.jpg", box=(20, 20, 40, 70))
        (coco / "labels").mkdir(parents=True, exist_ok=True)
        sd.write_labels(coco / "labels" / f"{stem}.txt", [(1, (20, 20, 40, 70))], 96, 96)
    (coco / "manifest.jsonl").write_text(json.dumps({"file": "coco_000000000000.jpg", "license": 4}) + "\n")
    props = tmp_path / "props"
    for i in range(10):
        p = props / "knife" / "hero-20261018-090000" / f"{i:04d}.jpg"
        _img(p, box=(30, 10, 50, 80))
        sd.write_labels(p.with_suffix(".txt"), [(1, (30, 10, 50, 80))], 96, 96)
    return coco, props


def test_build_refuses_unreviewed_test_photos_then_builds(tmp_path):
    coco, props = _fake_sources(tmp_path)
    out = tmp_path / "scout"
    args = ["build", "--coco", str(coco), "--props", str(props), "--out", str(out)]
    with pytest.raises(SystemExit, match="unreviewed"):
        bsd.main(args)
    for p in review_labels.pick(props, "test", only_unreviewed=True):
        review_labels.mark_reviewed(props, str(p.relative_to(props)))
    assert review_labels.status(props)["test"] == (2, 2)
    bsd.main(args)
    yaml = (out / "scout.yaml").read_text()
    assert "1: weapon" in yaml and "2: badge" in yaml
    rooms = {s: len(list((out / "images" / s).glob("room_*"))) for s in ("train", "val", "test")}
    assert rooms == {"train": 7, "val": 1, "test": 2}
    assert len(list(out.glob("images/*/coco_*"))) == 6
    for img in out.glob("images/*/*.jpg"):
        assert (out / "labels" / img.parent.name / f"{img.stem}.txt").exists()


def test_capture_props_from_a_named_synthetic_camera(tmp_path):
    saved = capture_props.main(["--camera", "hero", "--source", "synthetic", "--prop", "badge", "--n", "3",
                                "--interval", "0", "--countdown", "0", "--out", str(tmp_path)])
    assert [p.name for p in saved] == ["0000.jpg", "0001.jpg", "0002.jpg"]
    assert all(p.parent.parent.name == "badge" and cv2.imread(str(p)) is not None for p in saved)


def test_publish_is_a_dry_run_and_keeps_people_out(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # any upload attempt would fail loudly
    coco, props = _fake_sources(tmp_path)
    for p in sd.room_images(props):
        review_labels.mark_reviewed(props, str(p.relative_to(props)))
    bsd.main(["build", "--coco", str(coco), "--props", str(props), "--out", str(tmp_path / "scout")])
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    staged = publish_scout_hf.main(["--model-repo", "o/sherlock-scout", "--dataset-repo", "o/sherlock-scout-data",
                                    "--weights", str(weights), "--dataset", str(tmp_path / "scout"),
                                    "--coco-manifest", str(coco / "manifest.jsonl"),
                                    "--eval", str(tmp_path / "none.json"), "--out", str(tmp_path / "hf")])
    assert "nothing uploaded" in capsys.readouterr().out
    ds = staged["dataset"]
    assert not list(ds.glob("images/*/room_*")), "in-room photos show people: excluded by default"
    assert len(list(ds.glob("images/*/coco_*"))) == 6 and (ds / "manifest.jsonl").exists()
    assert "license: cc-by-4.0" in (ds / "README.md").read_text()
    card = (staged["model"] / "README.md").read_text()
    assert "license: agpl-3.0" in card and "o/sherlock-scout-data" in card


# --- tiny CPU train + eval ------------------------------------------------------------
@pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None, reason="ultralytics not installed")
def test_smoke_train_and_eval_on_cpu(tmp_path):
    ds = tmp_path / "scout"
    for split, n in (("train", 4), ("val", 2), ("test", 2)):
        for i in range(n):
            name = f"{'room_knife_s' if i % 2 else 'coco'}_{split}_{i}"
            _img(ds / "images" / split / f"{name}.jpg", box=(20, 10, 44, 80))
            (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
            sd.write_labels(ds / "labels" / split / f"{name}.txt", [(1, (20, 10, 44, 80))], 96, 96)
    data = ds / "scout.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(sd.CLASSES))
    data.write_text(f"path: {ds}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    # From the yaml, not yolov8n.pt: builds the net locally, so the test needs no weight download.
    best = train_scout.train(data, "yolov8n.yaml", project=tmp_path / "runs", plots=False,
                             **{**train_scout.SMOKE, "imgsz": 64})
    assert best.is_file()
    report = eval_scout.main(["--after", str(best), "--before", str(best), "--dataset", str(ds),
                              "--imgsz", "64", "--out", str(tmp_path / "eval.json")])
    assert set(report["images"]) == {"room", "coco"}
    assert {"room", "coco", "all"} <= set(report["after"]) and report["headline"].startswith("Scout weapon detection (room")
    assert json.loads((tmp_path / "eval.json").read_text())["split"] == "test"
