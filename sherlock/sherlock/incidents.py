"""Local incident manifests and explicitly simulated responder receipts.

Hashes detect changed image bytes against this local manifest, not an independent
or immutable chain of custody. No notification leaves this process.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import threading
import time

from . import signing

ID = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9]{3,}")
NAMES = {"original_frame", "model_context", "weapon_crop", "object_crop", "face_crop"}


class IncidentStore:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.lock = threading.RLock()

    def path(self, iid: str, filename: str = "manifest.json") -> pathlib.Path:
        if not ID.fullmatch(iid) or (filename not in {"manifest.json", "manifest.tmp", *(f"{n}.jpg" for n in NAMES), "model_context.png"} and not re.fullmatch(r"[a-f0-9]{64}\.(jpg|png)", filename)):
            raise ValueError("invalid evidence path")
        path = self.root / iid / filename
        if not path.resolve().is_relative_to(self.root.resolve()) or path.is_symlink() or path.parent.is_symlink():
            raise ValueError("invalid evidence path")
        return path

    def _history_path(self, iid: str, revision: int) -> pathlib.Path:
        folder = self.path(iid).parent / "revisions"
        if folder.is_symlink():
            raise ValueError("invalid revision path")
        path = folder / f"{revision:06d}.json"
        if path.is_symlink():
            raise ValueError("invalid revision path")
        return path

    def signed_history(self, iid: str) -> list[dict]:
        record = json.loads(self.path(iid).read_text())
        revision = record.get("signature", {}).get("revision", 0)
        if not isinstance(revision, int) or not 0 <= revision <= 10000:
            raise ValueError("invalid revision count")
        return [json.loads(self._history_path(iid, number).read_text())
                for number in range(1, revision + 1)]

    def _history_valid(self, record: dict) -> str:
        if "signature" not in record:
            return "unsigned"
        try:
            history = self.signed_history(record["incident"])
            previous = None
            fingerprint = record["signature"]["key_fingerprint"]
            for number, envelope in enumerate(history, 1):
                metadata = envelope["signature"]
                if (signing.verify(envelope) != "verified" or metadata["revision"] != number
                        or metadata["previous_sha256"] != previous
                        or metadata["key_fingerprint"] != fingerprint
                        or envelope["incident"] != record["incident"]):
                    return "invalid"
                previous = signing.digest(envelope)
            return "verified" if history and history[-1] == {**signing.payload(record), "signature": record["signature"]} else "invalid"
        except (OSError, ValueError, KeyError, TypeError):
            return "invalid"

    def _write(self, record: dict) -> None:
        path = self.path(record["incident"])
        prior = None
        if path.exists():
            prior = self.read(record["incident"])
            if prior["verification"] != {"signature": "verified", "artifacts": "verified", "history": "verified"}:
                raise ValueError("cannot update unsigned or damaged evidence")
        revision = prior["signature"]["revision"] + 1 if prior else 1
        previous = signing.digest({**signing.payload(prior), "signature": prior["signature"]}) if prior else None
        signed = signing.sign(record, self.root, revision, previous, time.time())
        if prior and signed["signature"]["key_fingerprint"] != prior["signature"]["key_fingerprint"]:
            raise ValueError("local signer changed; refusing revision")
        revision_path = self._history_path(record["incident"], revision)
        revision_path.parent.mkdir(parents=True, exist_ok=True)
        with revision_path.open("x") as stream:
            json.dump(signed, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        record.clear()
        record.update(signed)
        path = self.path(record["incident"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path(record["incident"], "manifest.tmp")
        with temporary.open("w") as stream:
            json.dump(record, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def save(self, report: dict, custody: list[dict], models: dict) -> dict:
        with self.lock:
            report = signing.payload(report)
            record = {**report, "custody": list(custody), "models": models, "artifacts": [],
                      "custody_scope": "Local application history; not independently attested or immutable"}
            record["custody"].append({"stage": "acquisition" if report.get("state") == "investigating" else "decision",
                "t": report["t"], "detail": "Investigation in progress; evidence saved before assessment"
                if report.get("state") == "investigating" else f"{report['action']}: {report.get('claim', '')}"})
            for name in report["evidence"]:
                filename = f"{name}.png" if name == "model_context" else f"{name}.jpg"
                path = self.path(report["incident"], filename)
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                stored_name = digest + path.suffix
                immutable = self.path(report["incident"], stored_name)
                if immutable.exists():
                    if immutable.read_bytes() != data:
                        raise ValueError("stored artifact hash mismatch")
                else:
                    with immutable.open("xb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                record["artifacts"].append({"name": name, "sha256": digest,
                    "url": f"/evidence/{report['incident']}/{stored_name}", "filename": stored_name, "integrity": "verified",
                    **report.get("evidence_provenance", {}).get(name, {"camera": None, "captured_at": None})})
                record["custody"].append({"stage": "evidence", "t": time.time(),
                    "detail": f"Saved {filename} locally; SHA-256 {digest}"})
            for name in report.get("evidence_unsaved", []):
                record["custody"].append({"stage": "evidence", "t": time.time(),
                    "detail": f"{name} was NOT saved; no stored artifact available"})
            camera = report.get("camera") or next((a.get("camera") for a in record["artifacts"] if a.get("camera")), None)
            mock = bool(models.get("mock"))
            areas = {"hero": "North lobby", "scout1": "East corridor", "scout2": "Loading entrance"}
            record["location"] = {"camera": camera, "building": "Building A" if mock else None,
                                  "floor": "Level 1" if mock else None,
                                  "area": areas.get(camera, "Demo camera area") if mock else None,
                                  "simulated": mock}
            record["response"] = {"status": "new", "simulated": True, "team_notified_at": None, "events": []}
            record["notification"] = {"simulated": True, "status": "not_sent",
                                      "delivered_at": None, "acknowledged_at": None}
            if report["action"] == "alert" and report.get("verified"):
                now = time.time()
                record["notification"].update(status="delivered", delivered_at=now)
                record["custody"].append({"stage": "notification", "t": now,
                    "detail": "Delivered to the local simulated phone; no external message sent"})
            self._write(record)
            return self.read(record["incident"])

    def read(self, iid: str) -> dict:
        with self.lock:
            record = json.loads(self.path(iid).read_text())
            if record.get("incident") != iid:
                raise ValueError("manifest identity mismatch")
            signature_status = signing.verify(record)
            history_status = self._history_valid(record)
            for artifact in record["artifacts"]:
                try:
                    digest = hashlib.sha256(self.path(iid, artifact.get("filename", artifact["name"] + (".png" if artifact["name"] == "model_context" else ".jpg"))).read_bytes()).hexdigest()
                    artifact["integrity"] = "verified" if digest == artifact["sha256"] else "mismatch"
                except (OSError, ValueError):
                    artifact["integrity"] = "missing"
            states = {a["integrity"] for a in record["artifacts"]}
            if history_status == "verified":
                for envelope in self.signed_history(iid):
                    for artifact in envelope.get("artifacts", []):
                        try:
                            data = self.path(iid, artifact["filename"]).read_bytes()
                            if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
                                states.add("mismatch")
                        except (OSError, ValueError, KeyError):
                            states.add("missing")
            record["verification"] = {"signature": signature_status,
                "artifacts": "mismatch" if "mismatch" in states else "missing" if "missing" in states else "verified",
                "history": history_status}
            return record

    def list(self, limit: int = 20, offset: int = 0, action: str | None = None,
             evidence_only: bool = False) -> list[dict]:
        records = []
        if not self.root.is_dir():
            return records
        for folder in sorted(self.root.iterdir(), key=lambda p: p.name, reverse=True):
            if not ID.fullmatch(folder.name):
                continue
            try:
                record = self.read(folder.name)
                if action and record.get("action") != action:
                    continue
                if evidence_only and not record.get("artifacts"):
                    continue
                records.append(record)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if len(records) >= offset + max(1, min(limit, 100)):
                break
        return records[offset:]

    def acknowledge(self, iid: str) -> dict:
        return self.respond(iid, "investigate")

    def respond(self, iid: str, action: str) -> dict:
        with self.lock:
            record = self.read(iid)
            if record["verification"] != {"signature": "verified", "artifacts": "verified", "history": "verified"}:
                raise ValueError("cannot respond to unsigned or damaged evidence")
            if record.get("action") != "alert" or not record.get("verified"):
                raise ValueError("incident has no eligible alert")
            if action not in {"investigate", "notify_team", "resolve"}:
                raise ValueError("invalid response action")
            response = record["response"]
            if any(event["action"] == action for event in response["events"]):
                return record
            if response["status"] == "resolved":
                raise ValueError("incident already resolved")
            if action != "investigate" and response["status"] != "investigating":
                raise ValueError("investigate before notifying or resolving")
            now = time.time()
            details = {"investigate": "Responder investigating on the simulated phone",
                       "notify_team": "Team notification simulated locally; no external message sent",
                       "resolve": "Responder resolved the simulated incident"}
            if action == "investigate":
                response["status"] = "investigating"
                record["notification"].update(status="acknowledged", acknowledged_at=now)
            elif action == "notify_team":
                response["team_notified_at"] = now
            else:
                response["status"] = "resolved"
            response["events"].append({"action": action, "t": now, "simulated": True})
            record["custody"].append({"stage": "acknowledgment" if action == "investigate" else "response",
                                      "t": now, "detail": details[action]})
            self._write(record)
            return self.read(iid)
