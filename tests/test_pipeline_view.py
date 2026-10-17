"""The /pipeline page is a pure consumer of the existing SSE + MJPEG endpoints,
so the test only checks that it mounts next to make_app() and points at them."""
import json
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sherlock.dashboard.pipeline import mount_pipeline


def _page_cfg(body: str) -> dict:
    m = re.search(r"const CFG = (\{.*?\});", body)
    assert m, "page config was not injected"
    return json.loads(m.group(1))


def test_pipeline_page_mounts_and_uses_existing_streams():
    app = mount_pipeline(FastAPI())
    r = TestClient(app).get("/pipeline")
    assert r.status_code == 200
    body = r.text
    assert "text/html" in r.headers["content-type"]
    for needle in ("/events", "/stream/wide", "/stream/crop", "/demo/fail/1", "/demo/cover/hero/"):
        assert needle in body, needle
    # the judge-facing promise: the boundary is drawn and only crops cross it
    assert "crops only" in body
    # the template token is always replaced, never served raw
    assert "__CFG_JSON__" not in body


def test_flow_page_is_the_default_and_topology_is_served():
    client = TestClient(mount_pipeline(FastAPI()))
    assert "Incident record" in client.get("/").text
    body = client.get("/topology").text
    assert "__CFG_JSON__" not in body and "What runs where" in body
    for needle in ("/events", 'href="/"', 'href="/log"'):
        assert needle in body, needle


def test_page_config_names_models_and_where_they_run(monkeypatch):
    for key in ("SHERLOCK_MOCK", "BREV_HOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DIRECTOR_BASE_URL", "https://gpu.brev.example/v1")
    monkeypatch.setenv("DIRECTOR_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    monkeypatch.setenv("ANALYST_BASE_URL", "http://localhost:8002/v1")
    monkeypatch.setenv("CAMERAS", "hero:insta360:name:Link,scout1:fake:1")
    cfg = _page_cfg(TestClient(mount_pipeline(FastAPI())).get("/topology").text)
    assert cfg["models"]["scout"]["where"] == "laptop"
    assert cfg["models"]["director"] == {"model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                                         "where": "gpu.brev.example", "url": "https://gpu.brev.example/v1"}
    assert cfg["models"]["analyst"]["where"] == "laptop"          # localhost, no tunnel declared
    assert [c["backend"] for c in cfg["cameras"]] == ["insta360", "fake"]
    monkeypatch.setenv("BREV_HOST", "brev-box")                  # a tunnel: localhost is really the box
    cfg = _page_cfg(TestClient(mount_pipeline(FastAPI())).get("/topology").text)
    assert cfg["models"]["analyst"]["where"] == "brev-box"


def test_fonts_are_vendored_so_the_page_works_offline():
    client = TestClient(mount_pipeline(FastAPI()))
    body = client.get("/pipeline").text
    assert "fonts.googleapis.com" not in body and "fonts.gstatic.com" not in body
    urls = re.findall(r"url\((/pipeline/fonts/[^)]+)\)", body)
    assert len(urls) == 3
    for url in urls:
        r = client.get(url)
        assert r.status_code == 200, url
        assert r.headers["content-type"] == "font/woff2"
        assert r.content[:4] == b"wOF2"


@pytest.mark.parametrize("name", ["missing.woff2", "OFL-ibm-plex-mono.txt", "..%2Fpipeline.html", "%2e%2e%2fserver.py"])
def test_font_route_serves_only_the_vendored_woff2_files(name):
    assert TestClient(mount_pipeline(FastAPI())).get(f"/pipeline/fonts/{name}").status_code == 404


@pytest.mark.parametrize(("env", "expected"), [
    ({}, {"mock": False}),
    ({"SHERLOCK_MOCK": "1"}, {"mock": True}),
    ({"SHERLOCK_EXPANSO": "connected"}, {"mock": False}),
])
def test_page_only_claims_what_is_running(monkeypatch, env, expected):
    """Mock mode is stated, and no environment flag can declare Expanso connected:
    the footer polls /expanso, which reports only real deliveries."""
    for key in ("SHERLOCK_MOCK", "SHERLOCK_EXPANSO"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    r = TestClient(mount_pipeline(FastAPI())).get("/pipeline")
    cfg = _page_cfg(r.text)
    assert {k: cfg[k] for k in expected} == expected
    assert "expanso" not in cfg
    assert "pollExpanso(" in r.text
