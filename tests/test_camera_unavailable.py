"""A camera that will not open is left out, loudly, and the demo runs on the rest.
Sherlock stops only when no camera opens, and then with one line, not a traceback."""
import runpy
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sherlock.camera import CameraError
from sherlock.config import CameraSpec, Config
from sherlock.dashboard.server import make_app
from sherlock.orchestrator import Sherlock

from sherlock import orchestrator

# A video file that does not exist: the same open failure as an unplugged camera
# or a wrong index, on any machine, without touching real camera hardware.
MISSING = str(Path(__file__).parent / "no-such-camera.mp4")


def cameras(*specs: tuple[str, str]) -> Config:
    return Config(mock=True, cameras=[CameraSpec(name, "fake", source) for name, source in specs])


def test_unopenable_camera_is_left_out_and_announced(tmp_path, capsys):
    s = Sherlock(cameras(("hero", "synthetic"), ("scout1", MISSING)), evidence_dir=str(tmp_path))

    assert list(s.cameras) == ["hero"]
    assert list(s.tools.cameras) == ["hero"]  # the Director never sees it
    assert "cannot open video source" in s.unavailable["scout1"]
    assert "scout1 unavailable: cannot open video source" in capsys.readouterr().out

    s.start()
    s.stop()
    assert "started with cameras ['hero'], unavailable ['scout1']" in capsys.readouterr().out


def test_no_camera_opened_stops_with_every_reason(tmp_path):
    with pytest.raises(CameraError, match=r"no camera opened \(hero: .*; scout1: .*\); check CAMERAS"):
        Sherlock(cameras(("hero", MISSING), ("scout1", MISSING)), evidence_dir=str(tmp_path))


def test_investigation_finishes_on_the_cameras_that_opened(tmp_path, monkeypatch):
    real = orchestrator.build_camera

    def build(name, backend, source):
        if name == "scout1":
            raise CameraError("cannot open video source '7'")
        return real(name, backend, source)

    monkeypatch.setattr(orchestrator, "build_camera", build)
    s = Sherlock(cameras(("hero", "0"), ("scout1", "7")), synthetic=True, evidence_dir=str(tmp_path))
    s.start()
    try:
        deadline = time.time() + 25
        while time.time() < deadline and not any(r.get("action") == "alert" for r in s.log):
            time.sleep(0.1)
    finally:
        s.stop()
    alert = next((r for r in s.log if r.get("action") == "alert"), None)
    assert alert and alert["verified"] is True
    assert alert["evidence"] == ["original_frame", "weapon_crop", "face_crop"] and alert["evidence_camera"] == "hero"


def test_dashboard_marks_the_camera_and_refuses_to_cover_it(tmp_path):
    s = Sherlock(cameras(("hero", "synthetic"), ("scout1", MISSING)), evidence_dir=str(tmp_path))
    client = TestClient(make_app(s))

    body = client.get("/cameras").json()
    assert [c["name"] for c in body["cameras"]] == ["hero"]
    assert [u["name"] for u in body["unavailable"]] == ["scout1"]
    assert "cannot open video source" in body["unavailable"][0]["error"]

    assert client.post("/demo/cover/scout1/on").status_code == 404
    assert client.post("/demo/cover/hero/on").status_code == 200
    assert s.cameras["hero"].covered is True


def test_cli_exits_with_one_line_when_no_camera_opens(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "--mock", "--port", "8123"])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(orchestrator, "Sherlock",
                        Mock(side_effect=CameraError("no camera opened (hero: x); check CAMERAS")))

    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "run.py"), run_name="__main__")

    assert stop.value.code == "Sherlock: no camera opened (hero: x); check CAMERAS"
