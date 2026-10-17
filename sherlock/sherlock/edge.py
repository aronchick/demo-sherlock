"""Hand events and reports to the Expanso Edge pipelines on this laptop.

Off unless SHERLOCK_EDGE=1. When on, two background threads POST to the local
Edge jobs (jobs/sherlock-*.yaml); nothing here can slow the camera loop:
the bus already drops events for a subscriber that falls behind, and every
POST has a short timeout and is dropped on failure.

- E1 telemetry: every bus event, sent in small JSON-array batches.
- E2 incidents: one message per report, with the complete saved evidence package.
  Scene, face and exact model-input images are evidence, not training examples.
- E3 examples (SHERLOCK_EDGE_EXAMPLES=1 only): crops the Verifier did not
  confirm, for the next Scout fine-tune. Never faces.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import queue
import threading
import time
import urllib.error
import urllib.request

from .events import BUS, Event
from .incidents import ID

LIVE_SECONDS = 30.0  # a delivery this recent means the Edge pipeline is carrying data
_BATCH_MAX = 50
_BATCH_WAIT = 1.0
_SENDABLE_CROPS = ("weapon_crop", "object_crop")
_EVIDENCE_NAMES = {*_SENDABLE_CROPS, "face_crop", "original_frame", "model_context"}


def _on(name: str) -> bool:
    return os.environ.get(name, "0").strip() == "1"


class EdgeSink:
    def __init__(self, evidence_dir: pathlib.Path, events_url: str, incidents_url: str,
                 examples_url: str | None = None, timeout: float = 5.0):
        self.evidence_dir = pathlib.Path(evidence_dir)
        self.events_url = events_url
        self.incidents_url = incidents_url
        self.examples_url = examples_url
        self.timeout = timeout
        self.delivered_at: float | None = None
        self.sent = {"events": 0, "incidents": 0, "examples": 0, "failed": 0}
        self._reports: queue.Queue = queue.Queue(maxsize=100)
        self._running = False
        self._threads: list[threading.Thread] = []

    @classmethod
    def from_env(cls, evidence_dir: pathlib.Path) -> EdgeSink | None:
        if not _on("SHERLOCK_EDGE"):
            return None
        host = os.environ.get("SHERLOCK_EDGE_HOST", "127.0.0.1").strip() or "127.0.0.1"
        return cls(
            evidence_dir,
            os.environ.get("SHERLOCK_EDGE_EVENTS_URL", f"http://{host}:23111/events"),
            os.environ.get("SHERLOCK_EDGE_INCIDENTS_URL", f"http://{host}:23112/incident"),
            os.environ.get("SHERLOCK_EDGE_EXAMPLES_URL", f"http://{host}:23113/example")
            if _on("SHERLOCK_EDGE_EXAMPLES") else None,
        )

    # --- lifecycle --------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        events = BUS.subscribe()
        for target, args in ((self._pump_events, (events,)), (self._pump_reports, ())):
            t = threading.Thread(target=target, args=args, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=self.timeout + 2)
        self._threads.clear()

    def status(self) -> dict:
        age = None if self.delivered_at is None else time.time() - self.delivered_at
        return {"edge": age is not None and age < LIVE_SECONDS,
                "last_delivery_s": None if age is None else round(age, 1),
                "sent": dict(self.sent), "examples": self.examples_url is not None}

    # --- transport --------------------------------------------------------------
    def _post(self, url: str, body: object, kind: str, n: int = 1) -> bool:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                ok = 200 <= r.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            ok = False
        if ok:
            self.delivered_at = time.time()
            self.sent[kind] += n
        else:
            self.sent["failed"] += n
        return ok

    # --- E1: telemetry ----------------------------------------------------------
    def _pump_events(self, events: queue.Queue) -> None:
        batch: list[dict] = []
        first = 0.0
        while self._running:
            try:
                ev: Event = events.get(timeout=0.25)
            except queue.Empty:
                ev = None
            if ev is not None:
                if not batch:
                    first = time.monotonic()
                batch.append(telemetry_record(ev))
                if is_report(ev):
                    try:
                        self._reports.put_nowait(ev.data)
                    except queue.Full:
                        self.sent["failed"] += 1
            if batch and (len(batch) >= _BATCH_MAX or time.monotonic() - first >= _BATCH_WAIT):
                self._post(self.events_url, batch, "events", len(batch))
                batch = []

    # --- E2 and E3: reports -----------------------------------------------------
    def _pump_reports(self) -> None:
        while self._running:
            try:
                report = self._reports.get(timeout=0.25)
            except queue.Empty:
                continue
            self._post(self.incidents_url, incident_record(report, self.evidence_dir), "incidents")
            if self.examples_url:
                example = example_record(report, self.evidence_dir)
                if example:
                    self._post(self.examples_url, example, "examples")


def is_report(ev: Event) -> bool:
    return ev.stage == "system" and "incident" in ev.data and "action" in ev.data


def telemetry_record(ev: Event) -> dict:
    return {"stage": ev.stage, "msg": ev.msg, "latency_ms": ev.latency_ms, "t": ev.t, "data": ev.data}


def _evidence_file(evidence_dir: pathlib.Path, incident: str, filename: str) -> pathlib.Path:
    if not ID.fullmatch(incident) or filename not in {
        "manifest.json", "model_context.png",
        *(f"{name}.jpg" for name in _EVIDENCE_NAMES if name != "model_context")
    }:
        raise ValueError("invalid evidence path")
    root = pathlib.Path(evidence_dir).resolve()
    folder = root / incident
    path = folder / filename
    if folder.is_symlink() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("invalid evidence path")
    return path


def _crop_file(evidence_dir: pathlib.Path, incident: str, name: str) -> pathlib.Path:
    extension = "png" if name == "model_context" else "jpg"
    return _evidence_file(evidence_dir, incident, f"{name}.{extension}")


def incident_record(report: dict, evidence_dir: pathlib.Path) -> dict:
    """Send saved scene/object/face evidence regardless of the incident decision.

    Only known manifest metadata and named evidence files are read. Local paths
    are provenance, never instructions to read an arbitrary file from a manifest.
    Transport acceptance does not attest downstream persistence.
    """
    incident = str(report.get("incident", ""))
    source = report
    try:
        manifest = json.loads(_evidence_file(evidence_dir, incident, "manifest.json").read_text())
        if isinstance(manifest, dict) and manifest.get("incident") == incident:
            source = manifest
    except (OSError, ValueError):
        pass
    crops: dict[str, dict] = {}
    provenance = source.get("evidence_provenance") or {}
    for name in source.get("evidence") or []:
        if not isinstance(name, str) or name not in _EVIDENCE_NAMES:
            continue
        try:
            path = _crop_file(evidence_dir, incident, name)
            raw = path.read_bytes()
        except (OSError, ValueError):
            continue
        entry = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "path": str(path),
                 "b64": base64.b64encode(raw).decode(),
                 "content_type": "image/png" if name == "model_context" else "image/jpeg"}
        if isinstance(provenance, dict) and isinstance(provenance.get(name), dict):
            entry["provenance"] = provenance[name]
        crops[name] = entry
    keep = ("incident", "action", "claim", "confidence", "verified", "verifier_reason", "description",
            "downgraded", "vlm_calls", "total_ms", "t", "camera", "label", "threat", "evidence_camera",
            "evidence", "evidence_unsaved", "evidence_provenance", "artifacts", "custody", "models",
            "custody_scope", "notification", "trigger", "comparison", "analyst_mismatch")
    out = {k: source[k] for k in keep if k in source}
    out["crops"] = crops
    return out


def example_record(report: dict, evidence_dir: pathlib.Path) -> dict | None:
    """A crop the Verifier saw and did not confirm: a hard example for the Scout."""
    if report.get("verified") is not False:
        return None
    incident = str(report.get("incident", ""))
    for name in _SENDABLE_CROPS:
        if name in (report.get("evidence") or []):
            try:
                raw = _crop_file(evidence_dir, incident, name).read_bytes()
            except (OSError, ValueError):
                return None
            return {"incident": incident, "crop": name, "label": report.get("label"),
                    "claim": report.get("claim"), "verifier_reason": report.get("verifier_reason"),
                    "b64": base64.b64encode(raw).decode()}
    return None
