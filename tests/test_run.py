"""The real CLI must keep camera feeds on loopback unless explicitly overridden."""

import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sherlock.dashboard import server

from sherlock import orchestrator


@pytest.mark.parametrize(
    "host, expected",
    [(None, "127.0.0.1"), ("", "127.0.0.1"), (" \t", "127.0.0.1"),
     ("0.0.0.0", "0.0.0.0"), ("::1", "::1")],
)
def test_dashboard_cli_bind_host(monkeypatch, host, expected):
    if host is None:
        monkeypatch.delenv("DASHBOARD_HOST", raising=False)
    else:
        monkeypatch.setenv("DASHBOARD_HOST", host)
    monkeypatch.setattr(sys, "argv", ["run.py", "--mock", "--synthetic", "--port", "8123"])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    sherlock = Mock()
    create_sherlock = Mock(return_value=sherlock)
    app = FastAPI()
    make_app = Mock(return_value=app)
    serve = Mock()
    monkeypatch.setattr(orchestrator, "Sherlock", create_sherlock)
    monkeypatch.setattr(server, "make_app", make_app)
    monkeypatch.setattr(uvicorn, "run", serve)

    runpy.run_path(str(Path(__file__).resolve().parents[1] / "run.py"), run_name="__main__")

    sherlock.start.assert_called_once_with()
    make_app.assert_called_once_with(sherlock)
    serve.assert_called_once_with(app, host=expected, port=8123, log_level="warning")
    assert TestClient(app).get("/pipeline").status_code == 200  # the data-flow view is on the served app
