"""Independent public-key verification and failure-closed response transitions."""
import base64
import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi.testclient import TestClient
from sherlock.config import Config
from sherlock.dashboard.server import make_app
from sherlock.incidents import IncidentStore
from sherlock.orchestrator import Sherlock
from sherlock.signing import canonical, digest, payload, verify

IID = "20261017-120000-001"


def save(store, content=b"first image", camera="hero", mock=True):
    path = store.path(IID, "face_crop.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return store.save({"incident": IID, "action": "alert", "verified": True,
                       "claim": "knife", "t": 10, "evidence": ["face_crop"],
                       "evidence_provenance": {"face_crop": {"camera": camera}}}, [], {"mock": mock})


def test_public_signature_and_persisted_key(tmp_path):
    record = save(IncidentStore(tmp_path))
    metadata = dict(record["signature"])
    signature = base64.b64decode(metadata.pop("signature"))
    public = base64.b64decode(metadata["public_key"])
    Ed25519PublicKey.from_public_bytes(public).verify(signature, canonical(metadata))
    assert metadata["payload_sha256"] == hashlib.sha256(canonical(payload(record))).hexdigest()
    assert (tmp_path / ".signing-key").stat().st_mode & 0o777 == 0o600
    updated = IncidentStore(tmp_path).respond(IID, "investigate")
    assert updated["signature"]["key_fingerprint"] == metadata["key_fingerprint"]
    assert updated["signature"]["revision"] == 2
    assert updated["verification"] == {"signature": "verified", "history": "verified", "artifacts": "verified"}


@pytest.mark.parametrize("field", ["claim", "location", "custody", "models", "notification"])
def test_manifest_tamper_blocks_resigning(tmp_path, field):
    store = IncidentStore(tmp_path)
    save(store)
    raw = json.loads(store.path(IID).read_text())
    raw[field] = "tampered"
    store.path(IID).write_text(json.dumps(raw))
    assert store.read(IID)["verification"]["signature"] == "invalid"
    with pytest.raises(ValueError, match="damaged"):
        store.respond(IID, "investigate")
    with pytest.raises(ValueError, match="damaged"):
        save(store)


def test_artifact_tamper_distinct_from_signature(tmp_path):
    store = IncidentStore(tmp_path)
    record = save(store)
    store.path(IID, record["artifacts"][0]["filename"]).write_bytes(b"tampered")
    result = store.read(IID)
    assert result["verification"]["signature"] == "verified"
    assert result["verification"]["artifacts"] == "mismatch"
    with pytest.raises(ValueError, match="damaged"):
        store.respond(IID, "investigate")


def test_immutable_captures_and_revision_chain(tmp_path):
    store = IncidentStore(tmp_path)
    first = save(store)
    second = save(store, b"subsequent PTZ frame")
    assert first["artifacts"][0]["filename"] != second["artifacts"][0]["filename"]
    assert store.path(IID, first["artifacts"][0]["filename"]).read_bytes() == b"first image"
    history = store.signed_history(IID)
    assert len(history) == 2
    assert all(verify(envelope) == "verified" for envelope in history)
    assert history[1]["signature"]["previous_sha256"] == digest(history[0])
    prior = store._history_path(IID, 1)
    prior.write_text(prior.read_text().replace('"knife"', '"cup"'))
    assert store.read(IID)["verification"]["history"] == "invalid"
    with pytest.raises(ValueError, match="damaged"):
        store.respond(IID, "investigate")


def test_unsigned_legacy_remains_unsigned(tmp_path):
    store = IncidentStore(tmp_path)
    record = save(store)
    store.path(IID).write_text(json.dumps(payload(record)))
    assert store.read(IID)["verification"]["signature"] == "unsigned"
    with pytest.raises(ValueError, match="unsigned"):
        store.acknowledge(IID)


def test_transitions_idempotence_and_location(tmp_path):
    store = IncidentStore(tmp_path)
    record = save(store)
    assert record["location"]["area"] == "North lobby"
    assert record["location"]["simulated"] is True
    with pytest.raises(ValueError, match="investigate before"):
        store.respond(IID, "notify_team")
    investigating = store.respond(IID, "investigate")
    assert store.respond(IID, "investigate") == investigating
    notified = store.respond(IID, "notify_team")
    assert notified["response"]["team_notified_at"]
    assert store.respond(IID, "notify_team") == notified
    resolved = store.respond(IID, "resolve")
    assert resolved["response"]["status"] == "resolved"
    assert len(resolved["response"]["events"]) == 3
    assert store.respond(IID, "resolve") == resolved
    live = save(IncidentStore(tmp_path / "live"), camera="unmapped", mock=False)
    assert live["location"]["area"] is None
    assert live["location"]["building"] is None
    assert live["location"]["simulated"] is False


def test_response_and_history_api(tmp_path):
    sherlock = Sherlock(Config(), synthetic=True, evidence_dir=str(tmp_path))
    record = save(sherlock.incidents)
    client = TestClient(make_app(sherlock))
    assert client.get(record["artifacts"][0]["url"]).content == b"first image"
    assert client.post(f"/incidents/{IID}/respond", json={"action": "investigate"}).status_code == 200
    assert client.post(f"/incidents/{IID}/respond", json={"action": "notify_team"}).status_code == 200
    assert len(client.get(f"/incidents/{IID}/signed-history").json()) == 3
    assert client.post(f"/incidents/{IID}/respond", json={"action": "send_real_message"}).status_code == 409
    assert client.post(f"/incidents/{IID}/respond", json=[]).status_code == 409


def test_old_revision_image_tampering_blocks_update(tmp_path):
    store = IncidentStore(tmp_path)
    first = save(store)
    save(store, b"later frame")
    store.path(IID, first["artifacts"][0]["filename"]).write_bytes(b"tampered old frame")
    assert store.read(IID)["verification"]["artifacts"] == "mismatch"
    with pytest.raises(ValueError, match="damaged"):
        store.respond(IID, "investigate")


def test_deleted_signer_key_cannot_replace_existing_signatures(tmp_path):
    store = IncidentStore(tmp_path)
    save(store)
    (tmp_path / ".signing-key").unlink()
    with pytest.raises(ValueError, match="signer changed"):
        store.respond(IID, "investigate")
    assert store.read(IID)["signature"]["revision"] == 1


def test_private_key_symlink_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"do not read this")
    root = tmp_path / "evidence"
    root.mkdir()
    (root / ".signing-key").symlink_to(outside)
    with pytest.raises(OSError):
        save(IncidentStore(root))


@pytest.mark.parametrize("initial", [True, False])
@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_manifest_publication_recovers_after_restart(tmp_path, monkeypatch, initial, failure):
    from pathlib import Path

    store = IncidentStore(tmp_path)
    if not initial:
        save(store)
    original_open = Path.open
    original_replace = Path.replace

    def fail_open(path, *args, **kwargs):
        if failure == "write" and path.name == "manifest.tmp":
            raise OSError("injected manifest write failure")
        return original_open(path, *args, **kwargs)

    def fail_replace(path, *args, **kwargs):
        if failure == "replace" and path.name == "manifest.tmp":
            raise OSError("injected manifest replace failure")
        return original_replace(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_open)
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="injected"):
            save(store) if initial else store.respond(IID, "investigate")
    restarted = IncidentStore(tmp_path)
    result = save(restarted) if initial else restarted.respond(IID, "investigate")
    assert result["verification"] == {"signature": "verified", "history": "verified", "artifacts": "verified"}
    assert len(restarted.signed_history(IID)) == 2
    if not initial:
        assert len(result["response"]["events"]) == 1


def test_recovery_refuses_tampered_orphan(tmp_path, monkeypatch):
    store = IncidentStore(tmp_path)
    with monkeypatch.context() as patch:
        def fail(_record):
            raise OSError("injected")
        patch.setattr(store, "_replace_manifest", fail)
        with pytest.raises(OSError, match="injected"):
            save(store)
    orphan = store._history_path(IID, 1)
    orphan.write_text(orphan.read_text().replace('"knife"', '"cup"'))
    with pytest.raises(ValueError, match="invalid unpublished"):
        save(IncidentStore(tmp_path))
    assert not store.path(IID).exists()


def test_provenance_cannot_override_authoritative_artifact_fields(tmp_path):
    store = IncidentStore(tmp_path)
    record = save(store)
    report = payload(record)
    report["evidence_provenance"]["face_crop"].update({
        "name": "forged", "sha256": "0" * 64, "filename": "manifest.json",
        "url": "https://example.invalid", "integrity": "verified", "relationship": "subsequent_capture"})
    updated = store.save(report, [], {"mock": True})
    artifact = updated["artifacts"][0]
    assert artifact["name"] == "face_crop"
    assert artifact["sha256"] == hashlib.sha256(b"first image").hexdigest()
    assert artifact["filename"] == record["artifacts"][0]["filename"]
    assert artifact["url"] == record["artifacts"][0]["url"]
    assert artifact["relationship"] == "subsequent_capture"
