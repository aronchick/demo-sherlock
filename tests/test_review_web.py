"""Browser review page: test photos listed first, drafts served on the 0-1000 canvas,
a save writes the YOLO labels and reviewed.txt, and nothing outside the photos is served."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import review_web
import scout_data as sd


@pytest.fixture
def props(tmp_path):
    props = tmp_path / "props"
    for prop in ("knife", "none"):
        d = props / prop / "hero-1"
        d.mkdir(parents=True)
        for i in range(10):
            cv2.imwrite(str(d / f"{i:04d}.jpg"), np.zeros((108, 192, 3), np.uint8))
    # a Gemini draft for knife 0009: person plus knife, in YOLO cx cy w h
    (props / "knife" / "hero-1" / "0009.txt").write_text("0 0.5 0.5 0.4 0.8\n1 0.7 0.5 0.1 0.2\n")
    cache = tmp_path / "gemini_labels" / "knife" / "hero-1"
    cache.mkdir(parents=True)
    (cache / "0009.json").write_text("{}")
    return props


def _client(props):
    return TestClient(review_web.make_app(props, props.parent / "gemini_labels"))


def test_lists_test_photos_first_with_draft_and_review_state(props):
    d = _client(props).get("/api/photos").json()
    assert d["classes"] == list(sd.CLASSES)
    splits = [p["split"] for p in d["photos"]]
    assert splits == sorted(splits, key=review_web.SPLIT_ORDER.index) and splits[0] == "test"
    first = d["photos"][0]
    assert first == {"rel": "knife/hero-1/0008.jpg", "split": "test", "prop": "knife",
                     "reviewed": False, "gemini": False}
    assert d["photos"][1]["gemini"] is True


def test_labels_come_back_on_the_canvas_and_a_save_marks_reviewed(props):
    client = _client(props)
    rel = "knife/hero-1/0009.jpg"
    d = client.get(f"/api/labels/{rel}").json()
    assert d["default_class"] == d["prop_class"] == sd.CLASSES.index("weapon")
    assert d["gemini"] is True and d["reviewed"] is False
    assert [b["cls"] for b in d["labels"]] == [0, 1]
    assert d["labels"][1]["box"] == pytest.approx([650, 400, 750, 600])
    knife = {"cls": 1, "box": [600, 350, 760, 640]}
    assert client.post(f"/api/labels/{rel}", json={"labels": [knife]}).json() == {"reviewed": 1}
    assert sd.load_reviewed(props) == {rel}
    cls, *box = (props / "knife" / "hero-1" / "0009.txt").read_text().split()
    assert int(cls) == 1 and [float(v) for v in box] == pytest.approx([0.68, 0.495, 0.16, 0.29])
    assert client.get(f"/api/labels/{rel}").json()["reviewed"] is True
    none = client.get("/api/labels/none/hero-1/0000.jpg").json()
    assert none["prop_class"] is None and none["labels"] == []


def test_refuses_bad_boxes_and_paths_outside_the_photos(props):
    (props / "secret.jpg").write_bytes(b"x")
    client = _client(props)
    rel = "knife/hero-1/0009.jpg"
    assert client.post(f"/api/labels/{rel}", json={"labels": [{"cls": 7, "box": [0, 0, 9, 9]}]}).status_code == 422
    assert client.post("/api/labels/knife/hero-1/0042.jpg", json={"labels": []}).status_code == 404
    # %2F, not "../": the client normalises a bare "../" away before it reaches the route
    assert client.get("/photos/knife/hero-1/..%2F..%2Fsecret.jpg").status_code == 404
    assert client.get("/photos/secret.jpg").status_code == 404
    assert client.get("/fonts/..%2Freview_web.py").status_code == 404
    assert client.get(f"/photos/{rel}").headers["content-type"] == "image/jpeg"
    assert client.get("/").status_code == 200
    assert sd.load_reviewed(props) == set()
