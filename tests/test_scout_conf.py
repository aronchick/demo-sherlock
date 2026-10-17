"""SCOUT_CONF reaches actual YOLO prediction without changing the default."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sherlock.config import CameraSpec, Config
from sherlock.orchestrator import Sherlock


@pytest.mark.parametrize("value,expected", [(None, 0.35), ("0.10", 0.10), ("0", 0.0), ("1", 1.0)])
def test_scout_conf_reaches_predict(tmp_path, monkeypatch, value, expected):
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
