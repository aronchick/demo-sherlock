"""Gemini pre-labelling with a fake agy: box conversion, the YOLO draft it writes, the
cache that keeps reruns from resending photos, and skipped reviewed photos."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gemini_prelabel as gp
import scout_data as sd


def test_yolo_labels_convert_and_drop_bad_boxes():
    boxes = [{"label": "coke", "box_2d": [400, 700, 550, 770]},
             {"label": "person", "box_2d": [0, 0, 1000, 500]},
             {"label": "coke", "box_2d": [500, 700, 400, 770]},  # ymin > ymax
             {"label": "cap", "box_2d": [0, 0, 10, 1200]}]       # off the canvas
    labels = gp.yolo_labels(boxes)
    assert labels == [(sd.CLASSES.index("coke"), (700, 400, 770, 550)),
                      (sd.CLASSES.index("person"), (0, 0, 500, 1000))]


def _photos(props: Path) -> list[Path]:
    out = []
    for prop in ("coke", "fedora"):
        d = props / prop / "hero-1"
        d.mkdir(parents=True)
        for i in range(3):
            p = d / f"{i:04d}.jpg"
            cv2.imwrite(str(p), np.zeros((1080, 1920, 3), np.uint8))
            out.append(p)
    return out


@pytest.fixture
def fake_agy(monkeypatch):
    calls = []

    def run(cmd, cwd, **_):
        sent = sorted(f.name for f in Path(cwd).glob("img*.jpg"))
        calls.append(sent)
        assert all(cv2.imread(str(Path(cwd) / f)).shape[1] == gp.SEND_WIDTH for f in sent)
        images = [{"file": f, "boxes": [{"label": "coke", "box_2d": [400, 700, 550, 770]}]} for f in sent[:-1]]
        out = {"status": "SUCCESS", "structured_output": {"images": images}}  # last photo "missed"
        return subprocess.CompletedProcess(cmd, 0, json.dumps(out), "")

    monkeypatch.setattr(gp.subprocess, "run", run)
    return calls


def test_labels_photos_caches_answers_and_resends_only_misses(tmp_path, fake_agy):
    props = tmp_path / "props"
    photos = _photos(props)
    (props / sd.REVIEWED).write_text("fedora/hero-1/0000.jpg\n")
    done = gp.main(["--props", str(props), "--batch", "3", "--jobs", "2"])
    assert len(fake_agy) == 2 and sum(map(len, fake_agy)) == 5, "the reviewed photo is never sent"
    assert len(done) == 3, "each batch's last photo had no answer"
    first = props / "coke" / "hero-1" / "0000.txt"
    cls, *box = first.read_text().split()
    assert sd.CLASSES[int(cls)] == "coke"
    assert [float(v) for v in box] == pytest.approx([0.735, 0.475, 0.07, 0.15])
    assert (tmp_path / "gemini_labels" / "coke" / "hero-1" / "0000.json").exists()
    fake_agy.clear()
    gp.main(["--props", str(props), "--batch", "3"])
    assert sum(map(len, fake_agy)) == 2, "a rerun sends only the photos without an answer"
    assert not (props / "fedora" / "hero-1" / "0000.txt").exists(), "reviewed labels untouched"
    assert photos


def test_a_failed_batch_writes_nothing(tmp_path, monkeypatch):
    props = tmp_path / "props"
    _photos(props)
    monkeypatch.setattr(gp.subprocess, "run",
                        lambda cmd, **_: subprocess.CompletedProcess(cmd, 1, "", "quota exhausted"))
    assert gp.main(["--props", str(props)]) == {}
    assert not list(props.glob("*/*/*.txt")) and not (tmp_path / "gemini_labels").exists()
