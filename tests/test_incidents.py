"""Local custody persistence and simulated receipt behavior."""
import json

import pytest
from fastapi.testclient import TestClient
from sherlock.config import Config
from sherlock.dashboard.server import make_app
from sherlock.events import BUS
from sherlock.incidents import IncidentStore
from sherlock.orchestrator import Sherlock

IID = "20261017-120000-001"


def save(store, action="alert", verified=True):
    path = store.path(IID, "weapon_crop.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image bytes")
    return store.save({"incident": IID, "action": action, "verified": verified,
                       "claim": "knife", "t": 10, "evidence": ["weapon_crop"]},
                      [{"stage": "scout", "t": 1, "detail": "detected"}], {"mock": True})


def test_persistence_tamper_and_idempotent_ack(tmp_path):
    save(IncidentStore(tmp_path))
    store = IncidentStore(tmp_path)
    assert store.list()[0]["notification"]["status"] == "delivered"
    record = store.acknowledge(IID)
    assert store.acknowledge(IID) == record
    assert sum(e["stage"] == "acknowledgment" for e in record["custody"]) == 1
    store.path(IID, record["artifacts"][0]["filename"]).write_bytes(b"tampered")
    assert store.read(IID)["artifacts"][0]["integrity"] == "mismatch"
    store.path(IID, record["artifacts"][0]["filename"]).unlink()
    assert store.read(IID)["artifacts"][0]["integrity"] == "missing"


@pytest.mark.parametrize(("action", "verified"), [("ignore", False), ("log", True), ("alert", False)])
def test_only_verified_alert_delivers(tmp_path, action, verified):
    store = IncidentStore(tmp_path)
    assert save(store, action, verified)["notification"]["status"] == "not_sent"
    with pytest.raises(ValueError):
        store.acknowledge(IID)


def test_api_and_path_guards(tmp_path):
    sherlock = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    save(sherlock.incidents)
    client = TestClient(make_app(sherlock))
    assert client.get("/incidents?limit=101").status_code == 422
    assert client.get("/incidents").json()[0]["incident"] == IID
    assert client.get(f"/incidents/{IID}").json()["models"] == {"mock": True}
    assert client.post(f"/incidents/{IID}/acknowledge").json()["notification"]["status"] == "acknowledged"
    assert client.get("/incidents/bad").status_code == 404
    for iid, name in [("../escape", "manifest.json"), (IID, "../escape"), (IID, "other.jpg")]:
        with pytest.raises(ValueError):
            sherlock.incidents.path(iid, name)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "20261017-120000-002"
    linked.symlink_to(outside, target_is_directory=True)
    assert client.get(f"/evidence/{linked.name}/weapon_crop.jpg").status_code == 404


def test_capture_is_scoped_and_excludes_images():
    with BUS.capture() as events:
        BUS.emit("tool", "actual capture")
        BUS.emit("model_input", "preview", image="private bytes")
    BUS.emit("tool", "outside")
    assert [event["detail"] for event in events] == ["actual capture"]
    assert "private" not in json.dumps(events)


def test_filtered_pagination(tmp_path):
    store = IncidentStore(tmp_path)
    base = save(store)
    for i in range(2, 8):
        report = {**base, "incident": f"20261017-120000-{i:03d}", "evidence": [],
                  "action": "ignore", "verified": False}
        store.save(report, [], {})
    assert store.list(1, action="alert")[0]["incident"] == IID
    assert store.list(1, evidence_only=True)[0]["incident"] == IID
    assert store.list(1, offset=2)[0]["incident"].endswith("005")
    assert store.list(2, offset=7) == []


def test_provenance_and_missing_evidence_are_explicit(tmp_path):
    store = IncidentStore(tmp_path)
    record = save(store)
    record["evidence_provenance"] = {"weapon_crop": {"camera": "fallback", "captured_at": 5}}
    record["evidence_unsaved"] = ["face_crop"]
    saved = store.save(record, [{"stage": "scout", "t": 1, "detail": "detected"}], {})
    assert saved["artifacts"][0]["camera"] == "fallback"
    assert saved["artifacts"][0]["captured_at"] == 5
    assert [a["name"] for a in saved["artifacts"]] == ["weapon_crop"]
    assert any("NOT saved" in event["detail"] for event in saved["custody"])
    times = [event["t"] for event in saved["custody"]]
    assert times == sorted(times)
