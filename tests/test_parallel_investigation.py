"""Concurrent checks on the masked crop, the Verifier's decision, retained evidence and provenance."""
import threading
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from sherlock.dashboard.server import make_app
from sherlock.events import BUS
from test_weapon_flow import _frozen_sherlock, _knife_det


def test_parallel_checks_share_the_masked_crop_and_capture_worker_events(tmp_path):
    s = _frozen_sherlock(tmp_path)
    barrier = threading.Barrier(2)
    seen = []

    def assess(role, image):
        # Sequential execution breaks this barrier, so this verifies actual overlap.
        barrier.wait(timeout=5)
        assert set(s.tools.evidence) == {"original_frame", "weapon_crop", "face_crop"}
        pending = s.incidents.read(s.tools.incident["id"])
        assert pending["state"] == "investigating"
        assert pending["notification"]["status"] == "not_sent"
        for name in s.tools.evidence:
            assert (tmp_path / s.tools.incident["id"] / f"{name}.jpg").exists()
        seen.append(image)
        BUS.emit(role, "worker result")

    def describe(image, question):
        assess("analyst", image)
        return {"description": "A knife."}

    def check(image, claim):
        assess("verifier", image)
        return {"supported": True, "reason": "yes"}

    s.tools.analyst = SimpleNamespace(describe=describe)
    s.tools.verifier = SimpleNamespace(check=check)
    s._investigate(_knife_det())
    report = s.log[-1]
    assert report["action"] == "alert"
    assert len(seen) == 2 and seen[0] is seen[1]
    assert not seen[0].flags.writeable
    assert np.array_equal(seen[0], s.tools.evidence["weapon_crop"])
    for name in ("face_crop", "original_frame"):
        other = s.tools.evidence[name]
        assert seen[0].shape != other.shape or not np.array_equal(seen[0], other)
    assert "model_context" not in s.tools.evidence and not (tmp_path / report["incident"] / "model_context.png").exists()
    record = s.incidents.read(report["incident"])
    stages = [e["stage"] for e in record["custody"]]
    assert stages.index("comparison") > stages.index("analyst")
    assert stages.index("comparison") > stages.index("verifier")
    assert "model_input" not in stages
    BUS.emit("analyst", "outside investigation")
    assert not any(e["detail"] == "outside investigation" for e in record["custody"])


def test_verifier_failure_preserves_full_package_and_yields_uncertain(tmp_path):
    s = _frozen_sherlock(tmp_path)

    def broken(*args):
        raise RuntimeError("endpoint unavailable")

    s.tools.verifier = SimpleNamespace(check=broken)
    s._investigate(_knife_det())
    report = s.log[-1]
    assert report["action"] == "log" and not report["verified"]
    assert report["comparison"]["status"] == "uncertain"
    assert set(report["evidence"]) == {"original_frame", "weapon_crop", "face_crop"}
    assert report["custody_saved"]
    assert report["notification"]["status"] == "not_sent"


def test_analyst_failure_never_blocks_a_confirmed_alert(tmp_path):
    s = _frozen_sherlock(tmp_path)

    def broken(*args):
        raise RuntimeError("endpoint unavailable")

    s.tools.analyst = SimpleNamespace(describe=broken)
    s._investigate(_knife_det())
    report = s.log[-1]
    assert report["action"] == "alert" and report["verified"]
    assert report["comparison"]["status"] == "confirmed" and report["description"] == ""


def test_original_frame_timestamp_and_ptz_relationship_are_distinct(tmp_path):
    s = _frozen_sherlock(tmp_path)
    frame = s.cameras["hero"].wide_frame()
    s.latest_wide["hero"] = frame
    s.latest_wide_at["hero"] = 123.0
    s._investigate(_knife_det())
    report = s.log[-1]
    p = report["evidence_provenance"]
    assert np.array_equal(s.tools.evidence["original_frame"], frame)
    assert p["original_frame"]["captured_at"] == 123.0
    assert p["original_frame"]["relationship"] == "source_frame"
    assert p["weapon_crop"]["captured_at"] > 123.0
    assert p["weapon_crop"]["relationship"] == "subsequent_capture"
    assert p["weapon_crop"]["source_artifact"] == "original_frame"
    assert p["weapon_crop"]["bbox"] == list(_knife_det().bbox)
    client = TestClient(make_app(s))
    assert client.get(f"/evidence/{report['incident']}/original_frame.jpg").status_code == 200


def test_simulated_disagreement_is_once_and_mock_only(tmp_path):
    s = _frozen_sherlock(tmp_path)
    client = TestClient(make_app(s))
    assert client.post("/demo/disagree/1").json()["simulated"]
    assert client.post("/demo/disagree/2").status_code == 400
    s._investigate(_knife_det())
    simulated = s.log[-1]
    # The Analyst advises: its contradicting words are withheld, the Verifier's yes stands.
    assert simulated["action"] == "alert" and simulated["comparison"]["status"] == "confirmed"
    assert simulated["description"] == "" and "ceramic cup" in simulated["analyst_mismatch"]
    s._investigate(_knife_det())
    assert s.log[-1]["action"] == "alert" and "analyst_mismatch" not in s.log[-1]
    s.cfg.mock = False
    assert client.post("/demo/disagree/1").status_code == 409


def test_acquisition_failure_keeps_source_scene_and_face(tmp_path):
    s = _frozen_sherlock(tmp_path)

    def broken(det):
        raise RuntimeError("camera controller failed")

    s.director = SimpleNamespace(investigate=broken)
    s._investigate(_knife_det())
    report = s.log[-1]
    assert report["action"] == "log" and not report["verified"]
    assert report["comparison"]["status"] == "uncertain"
    assert set(report["evidence"]) == {"original_frame", "face_crop"}
    assert report["evidence_provenance"]["face_crop"]["relationship"] == "crop_of_source"
    record = s.incidents.read(report["incident"])
    assert record["notification"]["status"] == "not_sent"
    assert any(e["stage"] == "comparison" for e in record["custody"])


def test_unparseable_verifier_is_uncertain_and_logs(tmp_path):
    s = _frozen_sherlock(tmp_path)
    s.tools.analyst = SimpleNamespace(describe=lambda image, question: {"description": "A knife."})
    s.tools.verifier = SimpleNamespace(check=lambda *args: {
        "supported": False, "determinate": False, "reason": "Cannot tell"})
    s._investigate(_knife_det())
    report = s.log[-1]
    assert report["comparison"]["status"] == "uncertain"
    assert report["comparison"]["verifier_supports"] is None
    assert report["action"] == "log"
