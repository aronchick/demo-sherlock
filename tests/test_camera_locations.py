"""Operator locations follow the trigger camera and remain bound to saved evidence."""
import json

import pytest
from sherlock.config import Config
from sherlock.incidents import IncidentStore
from sherlock.orchestrator import Sherlock

IID = "20261018-020000-001"


def test_configured_location_reaches_signed_incident_and_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERA_LOCATIONS", json.dumps({
        "hero": {"building": "Test building", "floor": "Level 2", "area": "West room"},
        "scout1": {"building": "Different building"},
    }))
    app = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    saved = app.incidents.save({"incident": IID, "action": "alert", "verified": True,
        "claim": "knife", "t": 10, "evidence": [], "camera": "scout1",
        "trigger": {"camera": "hero"}}, [], {"mock": False})
    assert saved["location"] == {"camera": "hero", "building": "Test building",
        "floor": "Level 2", "area": "West room", "simulated": False}
    assert saved["verification"]["signature"] == "verified"
    reopened = IncidentStore(tmp_path, {"hero": {"area": "Changed room"}})
    response = reopened.respond(IID, "investigate")
    assert response["location"] == saved["location"]
    assert all(value == "verified" for value in response["verification"].values())
    path = reopened.path(IID)
    tampered = json.loads(path.read_text())
    tampered["location"]["area"] = "Altered room"
    path.write_text(json.dumps(tampered))
    assert reopened.read(IID)["verification"]["signature"] != "verified"


def test_missing_real_location_stays_unknown(tmp_path):
    store = IncidentStore(tmp_path, {"hero": {"area": "Known room"}})
    saved = store.save({"incident": IID, "action": "ignore", "t": 1, "evidence": [],
                       "trigger": {"camera": "scout1"}}, [], {"mock": False})
    assert saved["location"] == {"camera": "scout1", "building": None,
                                 "floor": None, "area": None, "simulated": False}


@pytest.mark.parametrize("value", ['[]', '{"missing":{"area":"Room"}}',
    '{"hero":null}', '{"hero":{"address":"Room"}}', '{"hero":{"area":4}}',
    '{"hero":{"area":" "}}', json.dumps({"hero": {"area": "x" * 121}})])
def test_invalid_location_configuration_fails_before_startup(value, monkeypatch):
    monkeypatch.setenv("CAMERA_LOCATIONS", value)
    with pytest.raises((ValueError, TypeError)):
        Config()
