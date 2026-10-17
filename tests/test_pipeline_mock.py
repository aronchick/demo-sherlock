"""End-to-end in mock mode: scripted synthetic scene -> scout -> rule director ->
tools -> mock analyst/verifier. Also checks the failure-injection + camera-fallback path."""
import time

from sherlock.config import Config
from sherlock.events import BUS
from sherlock.orchestrator import Sherlock


def wait_for(s, pred, timeout=25):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hits = [r for r in s.log if pred(r)]
        if hits:
            return hits[0]
        time.sleep(0.1)
    return None


def is_alert(r):
    return r.get("action") == "alert"


def test_cascade_produces_verified_alert(tmp_path):
    s = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    s.start()
    r = wait_for(s, is_alert)
    s.stop()
    assert r, "no alert happened"
    assert "knife" in r["claim"].lower()
    assert r["verified"] is True
    assert BUS.stats["frames"] > BUS.stats["vlm_calls"] * 5, "cascade should gate the VLM"


def test_tool_failure_recovers_on_other_camera(tmp_path):
    before = dict(BUS.stats)
    s = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    s.inject_failure(1)
    s.start()
    r = wait_for(s, is_alert)
    s.stop()
    assert r and r["done"] and r["evidence"] == ["weapon_crop", "face_crop"]
    assert r["trigger"]["camera"] == "hero" and r["evidence_camera"] == "scout1"
    assert s.tools.injected_failures == 0
    assert BUS.stats["tool_errors"] > before["tool_errors"]
    assert BUS.stats["recoveries"] > before["recoveries"]


def test_corrupted_args_are_rejected_and_recovered(tmp_path):
    before = dict(BUS.stats)
    s = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    s.inject_corruption(1)
    s.start()
    r = wait_for(s, is_alert)
    s.stop()
    assert r and r["verified"] is True
    assert s.tools.injected_corruptions == 0
    assert BUS.stats["tool_errors"] > before["tool_errors"]


def test_covered_lens_is_loud_not_silent(tmp_path):
    s = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    s.cover("hero", True)
    s.start()
    r = wait_for(s, is_alert)
    s.stop()
    assert r and r["trigger"]["camera"] == "scout1"
