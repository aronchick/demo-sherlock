"""The Expanso Edge sink and its dashboard surface: off by default, sends only the masked
object crop and metadata (never a face or the full frame), never blocks the bus, and the
footer claims only what is delivering."""
import base64
import hashlib
import http.server
import json
import pathlib
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient
from sherlock.dashboard import server as dashboard
from sherlock.dashboard.server import make_app
from sherlock.edge import EdgeSink, example_record, incident_record
from sherlock.events import BUS

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEAPON = b"\xff\xd8weapon-jpeg-bytes"
FACE = b"\xff\xd8face-jpeg-bytes"
SCENE = b"\xff\xd8full-frame-jpeg-bytes"


def _evidence(tmp_path, incident="20261017-120000-001",
              crops=(("original_frame", SCENE), ("weapon_crop", WEAPON), ("face_crop", FACE))):
    folder = tmp_path / incident
    folder.mkdir(parents=True)
    for name, raw in crops:
        extension = "png" if name == "model_context" else "jpg"
        (folder / f"{name}.{extension}").write_bytes(raw)
    return incident


def _report(incident, action="alert", verified=True, evidence=("original_frame", "weapon_crop", "face_crop")):
    return {"incident": incident, "action": action, "claim": "a knife", "verified": verified,
            "evidence": list(evidence), "evidence_camera": "hero", "label": "knife", "threat": "weapon"}


# --- configuration -----------------------------------------------------------------

def test_sink_is_off_unless_asked(monkeypatch, tmp_path):
    monkeypatch.delenv("SHERLOCK_EDGE", raising=False)
    assert EdgeSink.from_env(tmp_path) is None
    monkeypatch.setenv("SHERLOCK_EDGE", "1")
    monkeypatch.delenv("SHERLOCK_EDGE_EXAMPLES", raising=False)
    sink = EdgeSink.from_env(tmp_path)
    assert sink.events_url == "http://127.0.0.1:23111/events"
    assert sink.incidents_url == "http://127.0.0.1:23112/incident"
    assert sink.examples_url is None  # E3 stays off: room photos show real people
    monkeypatch.setenv("SHERLOCK_EDGE_EXAMPLES", "1")
    assert EdgeSink.from_env(tmp_path).examples_url == "http://127.0.0.1:23113/example"


# --- what leaves the laptop ----------------------------------------------------------

def _no_face_or_scene(rec: dict) -> bool:
    sent = json.dumps(rec)
    return all(base64.b64encode(raw).decode() not in sent for raw in (FACE, SCENE)) and \
        "face_crop.jpg" not in sent and "original_frame.jpg" not in sent


@pytest.mark.parametrize("action", ["alert", "log", "ignore"])
def test_incidents_carry_only_the_masked_crop(tmp_path, action):
    iid = _evidence(tmp_path)
    rec = incident_record(_report(iid, action=action), tmp_path)
    assert set(rec["crops"]) == {"weapon_crop"}
    weapon = rec["crops"]["weapon_crop"]
    assert base64.b64decode(weapon["b64"]) == WEAPON
    assert weapon["sha256"] == hashlib.sha256(WEAPON).hexdigest()
    assert rec["evidence"] == ["original_frame", "weapon_crop", "face_crop"], "names are metadata"
    assert _no_face_or_scene(rec)


def test_hard_examples_are_unconfirmed_object_crops_only(tmp_path):
    iid = _evidence(tmp_path)
    assert example_record(_report(iid, verified=True), tmp_path) is None
    assert example_record(_report(iid, verified=None), tmp_path) is None
    ex = example_record(_report(iid, action="log", verified=False), tmp_path)
    assert ex["crop"] == "weapon_crop" and base64.b64decode(ex["b64"]) == WEAPON
    only_face = _evidence(tmp_path, "20261017-120000-002", crops=(("face_crop", FACE),))
    assert example_record(_report(only_face, verified=False, evidence=("face_crop",)), tmp_path) is None


# --- the sink against a real HTTP endpoint -------------------------------------------

class _Capture(http.server.BaseHTTPRequestHandler):
    got: ClassVar[list] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).got.append((self.path, json.loads(body)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture
def capture():
    _Capture.got = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Capture)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", _Capture.got
    srv.shutdown()
    srv.server_close()


def _wait(pred, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_sink_batches_events_and_posts_reports(tmp_path, capture):
    base, got = capture
    iid = _evidence(tmp_path)
    sink = EdgeSink(tmp_path, f"{base}/events", f"{base}/incident", f"{base}/example", timeout=2)
    assert sink.status()["edge"] is False
    sink.start()
    try:
        BUS.emit("scout", "new knife on hero")
        BUS.emit("system", "REPORT: a knife (alert, verified)", 12.0, **_report(iid))
        assert _wait(lambda: any(p == "/incident" for p, _ in got))
        assert _wait(lambda: any(p == "/events" for p, _ in got))
    finally:
        sink.stop()
    batches = [b for p, b in got if p == "/events"]
    assert all(isinstance(b, list) for b in batches)
    events = [e for b in batches for e in b]
    assert any(e["stage"] == "scout" and e["msg"] == "new knife on hero" for e in events)
    incident = next(b for p, b in got if p == "/incident")
    assert incident["incident"] == iid and set(incident["crops"]) == {"weapon_crop"}
    assert base64.b64decode(incident["crops"]["weapon_crop"]["b64"]) == WEAPON and _no_face_or_scene(incident)
    assert not any(p == "/example" for p, _ in got)  # verified: not a hard example
    status = sink.status()
    assert status["edge"] is True and status["sent"]["incidents"] == 1 and status["sent"]["failed"] == 0


def test_an_unreachable_edge_never_blocks_the_bus(tmp_path):
    sink = EdgeSink(tmp_path, "http://127.0.0.1:9/events", "http://127.0.0.1:9/incident", timeout=0.5)
    sink.start()
    try:
        t0 = time.perf_counter()
        for i in range(2000):
            BUS.emit("tool", f"call {i}")
        assert time.perf_counter() - t0 < 5.0
        assert _wait(lambda: sink.sent["failed"] > 0)
        assert sink.status()["edge"] is False
    finally:
        sink.stop()


# --- dashboard ------------------------------------------------------------------------

def _app(edge=None):
    return make_app(SimpleNamespace(edge=edge, cfg=SimpleNamespace(mock=True)))


def test_expanso_status_without_a_sink_claims_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("SHERLOCK_RUNTIME_DIR", str(tmp_path))
    body = TestClient(_app()).get("/expanso").json()
    assert body["edge"] is False and body["mode"] is None and body["serving"] == {}


def test_expanso_mode_comes_from_the_runner(monkeypatch, tmp_path):
    monkeypatch.setenv("SHERLOCK_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "expanso").mkdir()
    (tmp_path / "expanso" / "mode").write_text("cloud\n")
    assert dashboard.expanso_mode() == "cloud"
    (tmp_path / "expanso" / "mode").write_text("anything else")
    assert dashboard.expanso_mode() is None


def test_serving_metrics_accepted_only_from_loopback():
    app = _app()
    rec = {"server": "director", "up": True, "kv_cache_usage": 0.125, "prefix_cache_queries": 400,
           "prefix_cache_hits": 300, "requests_running": 2, "requests_waiting": 1, "extra": "dropped"}
    remote = TestClient(app, client=("203.0.113.9", 5000))
    assert remote.post("/api/serving", json=rec).status_code == 403
    local = TestClient(app, client=("127.0.0.1", 5000))
    assert local.post("/api/serving", json={"server": "other"}).status_code == 400
    assert local.post("/api/serving", content=b"not json").status_code == 400
    assert local.post("/api/serving", content=b"{" + b" " * 9000 + b"}").status_code == 413
    assert local.post("/api/serving", json=rec).status_code == 200
    shown = local.get("/expanso").json()["serving"]["director"]
    assert shown["kv_cache_usage"] == 0.125 and "extra" not in shown


# --- the jobs themselves ----------------------------------------------------------------

@pytest.mark.skipif(shutil.which("expanso-edge") is None, reason="expanso-edge not installed")
def test_expanso_jobs_validate():
    jobs = sorted(str(p) for p in (ROOT / "jobs").glob("sherlock-*.yaml"))
    assert len(jobs) == 5
    r = subprocess.run(["expanso-edge", "validate", *jobs], check=False, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


def test_rejections_are_not_cleared_by_catch():
    """A `catch` clears the error flag, so a rejected message (a face crop sent to
    E3) would flow on to the core. Rejections must stay errored for reject_errored."""
    for path in (ROOT / "jobs").glob("sherlock-*.yaml"):
        text = path.read_text()
        if "throw(" in text and "reject_errored" in text:
            assert "catch:" not in text, path.name


def test_saved_manifest_metadata_travels_without_scene_or_face_bytes(tmp_path):
    legacy = b"\x89PNG\r\n\x1a\ncontext"  # a #83-era composite of scene, crop and face
    iid = _evidence(tmp_path, crops=(("original_frame", SCENE), ("weapon_crop", WEAPON), ("face_crop", FACE),
                                     ("model_context", legacy)))
    manifest = {**_report(iid, "log", False, ("original_frame", "weapon_crop", "face_crop", "model_context")),
                "custody": [{"stage": "capture"}], "models": {"analyst": "configured"},
                "evidence_provenance": {"weapon_crop": {"source_artifact": "original_frame", "bbox": [1, 2, 3, 4]}},
                "unrelated_secret": "not transported"}
    (tmp_path / iid / "manifest.json").write_text(json.dumps(manifest))
    rec = incident_record(_report(iid), tmp_path)
    assert rec["action"] == "log" and rec["custody"] == manifest["custody"]
    assert set(rec["crops"]) == {"weapon_crop"}
    assert rec["crops"]["weapon_crop"]["provenance"]["source_artifact"] == "original_frame"
    assert _no_face_or_scene(rec) and base64.b64encode(legacy).decode() not in json.dumps(rec)
    assert "unrelated_secret" not in rec


def test_incident_job_drops_every_image_but_the_masked_crop():
    """The second line, on the Edge: whatever a sender includes, only these crops travel."""
    text = (ROOT / "jobs" / "sherlock-incidents.yaml").read_text()
    assert 'filter(kv -> ["weapon_crop", "object_crop"].contains(kv.key))' in text


def test_transport_refuses_unsafe_artifact_reads(tmp_path):
    iid = _evidence(tmp_path, crops=())
    secret = tmp_path / "unrelated.txt"
    secret.write_bytes(b"private")
    (tmp_path / iid / "weapon_crop.jpg").symlink_to(secret)
    (tmp_path / iid / "face_crop.jpg").symlink_to(secret)
    report = _report(iid, evidence=("weapon_crop", "face_crop", "../unrelated", "unknown"))
    assert incident_record(report, tmp_path)["crops"] == {}
    assert incident_record(_report("../outside"), tmp_path)["crops"] == {}
    assert example_record(_report("../outside", verified=False), tmp_path) is None
