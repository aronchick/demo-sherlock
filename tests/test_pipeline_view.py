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
    ({}, {"mock": False, "expanso": False}),
    ({"SHERLOCK_MOCK": "1"}, {"mock": True, "expanso": False}),
    ({"SHERLOCK_EXPANSO": "connected"}, {"mock": False, "expanso": True}),
    ({"SHERLOCK_EXPANSO": "yes"}, {"mock": False, "expanso": False}),
])
def test_page_only_claims_what_is_running(monkeypatch, env, expected):
    """Expanso shows as connected only when explicitly declared, and mock mode is stated."""
    for key in ("SHERLOCK_MOCK", "SHERLOCK_EXPANSO"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    r = TestClient(mount_pipeline(FastAPI())).get("/pipeline")
    assert _page_cfg(r.text) == expected
