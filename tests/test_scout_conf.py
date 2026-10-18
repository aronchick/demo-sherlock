"""SCOUT_CONF reaches actual YOLO prediction without changing the default."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sherlock.agents.scout import YoloScout
from sherlock.config import CameraSpec, Config
from sherlock.orchestrator import Sherlock


@pytest.fixture(autouse=True)
def isolate_resolution_and_nms(monkeypatch):
    monkeypatch.delenv("SCOUT_IMGSZ", raising=False)
    monkeypatch.delenv("SCOUT_NMS_IOU", raising=False)


def test_resolution_and_nms_reach_actual_predict(monkeypatch):
    calls = []
    model = SimpleNamespace(predict=lambda frame, **kwargs:
                            calls.append(kwargs) or [SimpleNamespace(boxes=[])])
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: model))
    monkeypatch.setenv("SCOUT_IMGSZ", "960")
    monkeypatch.setenv("SCOUT_NMS_IOU", "0.6")
    detector = YoloScout()
    detector.detect(np.zeros((50, 50, 3)), "hero")
    assert calls[0]["imgsz"] == 960
    assert calls[0]["iou"] == .6


@pytest.mark.parametrize("key,value", [("SCOUT_IMGSZ", "0"), ("SCOUT_IMGSZ", "641"),
    ("SCOUT_IMGSZ", "4096"), ("SCOUT_NMS_IOU", "nan"), ("SCOUT_NMS_IOU", "0"),
    ("SCOUT_NMS_IOU", "1.1")])
def test_invalid_predict_settings_fail(monkeypatch, key, value):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: None))
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        YoloScout()


@pytest.mark.parametrize("value,expected", [(None, 0.35), ("0.10", 0.10), ("0", 0.0), ("1", 1.0)])
def test_scout_conf_reaches_predict(tmp_path, monkeypatch, value, expected):
    monkeypatch.delenv("SCOUT_WEAPON_CONF", raising=False)
    if value is None:
        monkeypatch.delenv("SCOUT_CONF", raising=False)
    else:
        monkeypatch.setenv("SCOUT_CONF", value)
    calls = []

    class FakeModel:
        def __init__(self, _path):
            pass

        def predict(self, _frame, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=[])]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    monkeypatch.delenv("SCOUT_MISLABEL", raising=False)
    cfg = Config(mock=False, scout_backend="yolo", cameras=[CameraSpec("hero", "fake", "synthetic")])
    app = Sherlock(cfg, evidence_dir=str(tmp_path))
    try:
        assert cfg.scout_conf == expected
        assert app.scout.impl.detect(np.zeros((32, 32, 3), dtype=np.uint8), "hero") == []
        assert calls == [{"conf": expected, "verbose": False}]
    finally:
        app.stop()


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf", "not-a-number"])
def test_invalid_scout_conf_is_rejected(monkeypatch, value):
    monkeypatch.setenv("SCOUT_CONF", value)
    with pytest.raises(ValueError):
        Config()


def test_weapon_cutoff_does_not_lower_the_benign_cutoff(monkeypatch):
    monkeypatch.setenv("SCOUT_WEAPON_CONF", "0.10")
    calls = []
    boxes = [SimpleNamespace(cls=c, conf=p, xyxy=np.array([[1, 2, 30, 40]]))
             for c, p in [(0, .148), (1, .148), (1, .5), (0, .05)]]

    class Model:
        def __init__(self, _):
            pass

        def predict(self, frame, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=boxes, names={0: "knife", 1: "cup"})]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    detections = YoloScout().detect(np.zeros((50, 50, 3)), "hero")
    assert [(d.label, d.conf) for d in detections] == [("knife", .148), ("cup", .5)]
    assert calls == [{"conf": .10, "verbose": False}]
