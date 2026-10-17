"""Scout fine-tune kit: class map, labels, COCO selection, splits, metrics, the
label editor, dataset build, capture, a dry-run publish, and a tiny CPU train+eval."""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from sherlock.dashboard.numbers import read_numbers, scout_result

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


def test_hat_props_label_as_their_own_classes():
    assert sd.CLASSES[:4] == ("person", "weapon", "badge", "bottle"), "appending keeps old label ids"
    assert sd.PROP_CLASS["fedora"] == "fedora" and sd.PROP_CLASS["cap"] == "cap"
    assert len(review_labels.COLORS) == len(sd.CLASSES), "every class needs a box colour in the reviewer"
    assert review_labels.prop_default_class("fedora/hero-1/0000.jpg") == "fedora"
    assert review_labels.prop_default_class("none/hero-1/0000.jpg") == "weapon"
    ed = review_labels.LabelEditor([], review_labels.prop_default_class("cap/hero-1/0000.jpg"))
    assert ed.add((10, 10), (40, 40)) and sd.CLASSES[ed.labels[0][0]] == "cap"


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


def measured_evaluation():
    return {
        "evidence_kind": "model_evaluation", "execution_mode": "measured", "split": "test",
        "images": {"room": 1}, "conf": 0.35, "iou": 0.5, "headline": "Untrusted headline",
        "before": {"model": "stock.pt", "room": {"weapon": {
            "tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5}}},
        "after": {"model": "tuned.pt", "room": {"weapon": {
            "tp": 2, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0}}},
    }


@pytest.mark.parametrize("with_evaluation", [False, True])
def test_publish_is_a_dry_run_and_keeps_people_out(tmp_path, monkeypatch, capsys, with_evaluation):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # any upload attempt would fail loudly
    coco, props = _fake_sources(tmp_path)
    for p in sd.room_images(props):
        review_labels.mark_reviewed(props, str(p.relative_to(props)))
    bsd.main(["build", "--coco", str(coco), "--props", str(props), "--out", str(tmp_path / "scout")])
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    eval_path = tmp_path / "eval.json"
    if with_evaluation:
        eval_path.write_text(json.dumps(measured_evaluation()))
    staged = publish_scout_hf.main(["--model-repo", "o/sherlock-scout", "--dataset-repo", "o/sherlock-scout-data",
                                    "--weights", str(weights), "--dataset", str(tmp_path / "scout"),
                                    "--coco-manifest", str(coco / "manifest.jsonl"),
                                    "--eval", str(eval_path), "--out", str(tmp_path / "hf")])
    assert "nothing uploaded" in capsys.readouterr().out
    ds = staged["dataset"]
    assert not list(ds.glob("images/*/room_*")), "in-room photos show people: excluded by default"
    assert len(list(ds.glob("images/*/coco_*"))) == 6 and (ds / "manifest.jsonl").exists()
    assert "license: cc-by-4.0" in (ds / "README.md").read_text()
    card = (staged["model"] / "README.md").read_text()
    assert "license: agpl-3.0" in card and "o/sherlock-scout-data" in card
    assert (staged["model"] / "eval.json").exists() is with_evaluation
    assert ("recall 50% -> 100%" in card) is with_evaluation
    assert "Untrusted headline" not in card


@pytest.mark.parametrize("evaluation", [
    {"evidence_kind": "model_evaluation", "execution_mode": "smoke"},
    {"evidence_kind": "model_evaluation"},
    {"execution_mode": "measured"},
    {"evidence_kind": "model_evaluation", "execution_mode": "measured"},
    {"evidence_kind": "model_evaluation", "execution_mode": "measured", "stub": True},
    {}, None, [],
])
def test_publish_rejects_unmeasured_evidence_before_replacing_staging(tmp_path, evaluation):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(evaluation))
    out = tmp_path / "staged"
    out.mkdir()
    previous = out / "previous.txt"
    previous.write_text("preserve this staging")
    with pytest.raises((TypeError, ValueError), match="cannot be published"):
        publish_scout_hf.stage(tmp_path / "weights.pt", tmp_path / "dataset", out,
                              "org/model", "org/data", "agpl-3.0", False, eval_path,
                              tmp_path / "manifest.jsonl")
    assert previous.read_text() == "preserve this staging"


@pytest.mark.parametrize("path", [
    ("split",), ("conf",), ("iou",), ("images",), ("before", "model"),
    ("after", "room", "weapon", "recall"), ("before", "room", "weapon", "fp"),
])
def test_publisher_requires_complete_measured_report(path):
    evaluation = measured_evaluation()
    parent = evaluation
    for key in path[:-1]:
        parent = parent[key]
    parent.pop(path[-1])
    with pytest.raises(ValueError, match="cannot be published"):
        publish_scout_hf.measured_headline(evaluation)


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("marker,value", [("execution_mode", "smoke"), ("mock", True)])
def test_publisher_rejects_nested_smoke_or_mock(side, marker, value):
    evaluation = measured_evaluation()
    evaluation[side][marker] = value
    with pytest.raises(ValueError, match="cannot be published"):
        publish_scout_hf.measured_headline(evaluation)


@pytest.mark.parametrize("args,name", [([], "scout"), (["--smoke"], "scout_smoke"),
                                      (["--smoke", "--name", "custom"], "custom")])
def test_train_keeps_smoke_weights_separate_by_default(tmp_path, monkeypatch, args, name):
    data = tmp_path / "scout.yaml"
    data.write_text("test fixture")
    measured = tmp_path / "runs/scout/weights/best.pt"
    measured.parent.mkdir(parents=True)
    measured.write_bytes(b"measured weights")

    def train(_data, _model, project, run_name, **_kwargs):
        assert run_name == name
        weights = project / run_name / "weights/best.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"new weights")
        return weights

    monkeypatch.setattr(train_scout, "train", train)
    best = train_scout.main(["--data", str(data), "--project", str(tmp_path / "runs"), *args])
    assert best == tmp_path / "runs" / name / "weights/best.pt"
    assert best.read_bytes() == b"new weights"
    if name != "scout":
        assert measured.read_bytes() == b"measured weights"


def test_train_picks_cuda_then_mps_then_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    for cuda, mps, want in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
        monkeypatch.setattr(torch.cuda, "is_available", lambda cuda=cuda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda mps=mps: mps)
        assert train_scout.pick_device() == want
    assert train_scout.PROFILES["mps"]["device"] == "mps" and train_scout.PROFILES["cuda"]["device"] == "0"


# --- evaluation evidence labels ------------------------------------------------------
@pytest.fixture
def eval_args(tmp_path, monkeypatch):
    ds = tmp_path / "scout"
    _img(ds / "images/test/room_knife.jpg", box=(20, 10, 44, 80))
    labels = ds / "labels/test/room_knife.txt"
    labels.parent.mkdir(parents=True)
    sd.write_labels(labels, [(1, (20, 10, 44, 80))], 96, 96)

    # Exercise the real reader/scorer/writer without optional Torch or downloads.
    def predict(model, images, _conf, _imgsz, _device):
        predictions = [("weapon", 0.9, (20, 10, 44, 80))] if model == "tuned.pt" else []
        return dict.fromkeys(images, predictions)

    monkeypatch.setattr(eval_scout, "predict", predict)
    monkeypatch.setattr(eval_scout, "ROOT", tmp_path)
    return ["--after", "tuned.pt", "--before", "stock.pt", "--dataset", str(ds)]


@pytest.mark.parametrize("smoke", [False, True])
def test_eval_labels_persist_and_control_dashboard_evidence(tmp_path, eval_args, smoke):
    output = tmp_path / "runs/scout/eval.json"
    report = eval_scout.main([*eval_args, "--out", str(output), "--smoke" if smoke else "--measured"])
    saved = json.loads(output.read_text())
    assert saved == report
    assert saved["evidence_kind"] == "model_evaluation"
    assert saved["execution_mode"] == ("smoke" if smoke else "measured")
    result = read_numbers(tmp_path)["scout"]
    if smoke:
        assert result is None
        with pytest.raises(ValueError, match="cannot be published"):
            publish_scout_hf.measured_headline(saved)
    else:
        assert result["before_recall"] == 0 and result["after_recall"] == 1
        assert "recall 0% -> 100%" in publish_scout_hf.measured_headline(saved)


def test_eval_requires_explicit_mode_before_reading_data(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_scout, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as error:
        eval_scout.main(["--after", "smoke-trained.pt"])
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_smoke_default_preserves_measured_evidence(tmp_path, eval_args):
    eval_scout.main([*eval_args, "--measured"])
    measured = tmp_path / "runs/scout/eval.json"
    saved = measured.read_bytes()
    eval_scout.main([*eval_args, "--smoke"])
    assert measured.read_bytes() == saved
    smoke = json.loads((tmp_path / "runs/scout/eval_smoke.json").read_text())
    assert smoke["execution_mode"] == "smoke"
    assert read_numbers(tmp_path)["scout"]["file"] == "eval.json"


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
                              "--smoke", "--imgsz", "64", "--out", str(tmp_path / "eval.json")])
    assert set(report["images"]) == {"room", "coco"}
    assert {"room", "coco", "all"} <= set(report["after"]) and report["headline"].startswith("Scout weapon detection (room")
    written = json.loads((tmp_path / "eval.json").read_text())
    assert written == report
    assert written["split"] == "test" and written["evidence_kind"] == "model_evaluation"
    assert written["execution_mode"] == "smoke"
    assert scout_result(written) is None


def test_baseline_only_measured_has_no_after(tmp_path, eval_args):
    args = eval_args[2:]
    report = eval_scout.main([*args, "--baseline-only", "--measured", "--device", "mps"])
    assert report["after"] is None
    assert report["evaluation_scope"] == "baseline_only"
    assert report["environment"] == "local" and report["device"] == "mps"
    assert not (tmp_path / "runs/scout/eval.json").exists()
    result = read_numbers(tmp_path)["scout"]
    assert result["before_precision"] == 0 and result["after_precision"] is None
    assert result["after_recall"] is None and result["device"] == "mps"
    with pytest.raises(ValueError, match="cannot be published"):
        publish_scout_hf.measured_headline(report)
