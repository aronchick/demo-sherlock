"""End-to-end in mock mode: synthetic video -> scout -> rule director -> tools ->
mock analyst/verifier. Also checks the failure-injection + camera-fallback path."""
import os
import time

os.environ["SHERLOCK_MOCK"] = "1"
os.environ["CAMERAS"] = "hero:fake:synthetic,scout1:fake:synthetic"
os.environ["DWELL_SECONDS"] = "0.3"

from sherlock.config import Config
from sherlock.events import BUS
from sherlock.orchestrator import Sherlock


def _wait_reports(s, n, timeout=25):
    t0 = time.time()
    while len(s.log) < n and time.time() - t0 < timeout:
        time.sleep(0.2)
    return s.log


def test_cascade_produces_verified_report():
    s = Sherlock(Config(), synthetic=True)
    s.start()
    log = _wait_reports(s, 1)
    s.running = False
    assert log, "no investigation happened"
    r = log[0]
    assert r["done"] and "ACME" in r["claim"] or "blue" in r["claim"].lower()
    assert r["verified"] is True
    assert BUS.stats["frames"] > BUS.stats["vlm_calls"] * 5, "cascade should gate the VLM"


def test_tool_failure_recovers_on_other_camera():
    s = Sherlock(Config(), synthetic=True)
    s.inject_failure(1)
    s.start()
    log = _wait_reports(s, 1)
    s.running = False
    assert log and log[0]["done"]
    assert BUS.stats["tool_errors"] >= 1 and BUS.stats["recoveries"] >= 1


def test_covered_lens_is_loud_not_silent():
    s = Sherlock(Config(), synthetic=True)
    s.cover("hero", True)
    s.start()
    log = _wait_reports(s, 1)
    s.running = False
    assert log and log[0]["trigger"]["camera"] == "scout1"
