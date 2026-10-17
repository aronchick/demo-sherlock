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

ID = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9]{3,}")
NAMES = {"original_frame", "model_context", "weapon_crop", "object_crop", "face_crop"}


class IncidentStore:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.lock = threading.RLock()

    def path(self, iid: str, filename: str = "manifest.json") -> pathlib.Path:
        if not ID.fullmatch(iid) or filename not in {"manifest.json", "manifest.tmp", *(f"{n}.jpg" for n in NAMES), "model_context.png"}:
            raise ValueError("invalid evidence path")
        path = self.root / iid / filename
        if not path.resolve().is_relative_to(self.root.resolve()) or path.is_symlink() or path.parent.is_symlink():
            raise ValueError("invalid evidence path")
        return path

    def _write(self, record: dict) -> None:
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
            record = {**report, "custody": list(custody), "models": models, "artifacts": [],
                      "custody_scope": "Local application history; not independently attested or immutable"}
            record["custody"].append({"stage": "acquisition" if report.get("state") == "investigating" else "decision",
                "t": report["t"], "detail": "Investigation in progress; evidence saved before assessment"
                if report.get("state") == "investigating" else f"{report['action']}: {report.get('claim', '')}"})
            for name in report["evidence"]:
                filename = f"{name}.png" if name == "model_context" else f"{name}.jpg"
                path = self.path(report["incident"], filename)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                record["artifacts"].append({"name": name, "sha256": digest,
                    "url": f"/evidence/{report['incident']}/{filename}", "integrity": "verified",
                    **report.get("evidence_provenance", {}).get(name, {"camera": None, "captured_at": None})})
                record["custody"].append({"stage": "evidence", "t": time.time(),
                    "detail": f"Saved {name}.jpg locally; SHA-256 {digest}"})
            for name in report.get("evidence_unsaved", []):
                record["custody"].append({"stage": "evidence", "t": time.time(),
                    "detail": f"{name}.jpg was NOT saved; no stored artifact available"})
            record["notification"] = {"simulated": True, "status": "not_sent",
                                      "delivered_at": None, "acknowledged_at": None}
            if report["action"] == "alert" and report.get("verified"):
                now = time.time()
                record["notification"].update(status="delivered", delivered_at=now)
                record["custody"].append({"stage": "notification", "t": now,
                    "detail": "Delivered to the local simulated phone; no external message sent"})
            self._write(record)
            return record

    def read(self, iid: str) -> dict:
        with self.lock:
            record = json.loads(self.path(iid).read_text())
            if record.get("incident") != iid:
                raise ValueError("manifest identity mismatch")
            for artifact in record["artifacts"]:
                try:
                    digest = hashlib.sha256(self.path(iid, artifact["name"] + (".png" if artifact["name"] == "model_context" else ".jpg")).read_bytes()).hexdigest()
                    artifact["integrity"] = "verified" if digest == artifact["sha256"] else "mismatch"
                except (OSError, ValueError):
                    artifact["integrity"] = "missing"
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
        with self.lock:
            record = self.read(iid)
            notification = record["notification"]
            if notification["status"] == "not_sent":
                raise ValueError("incident has no simulated notification")
            if notification["status"] != "acknowledged":
                now = time.time()
                notification.update(status="acknowledged", acknowledged_at=now)
                record["custody"].append({"stage": "acknowledgment", "t": now,
                    "detail": "Responder acknowledged on the simulated phone"})
                self._write(record)
            return record
