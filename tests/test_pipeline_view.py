"""The /pipeline page is a pure consumer of the existing SSE + MJPEG endpoints,
so the test only checks that it mounts next to make_app() and points at them."""
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sherlock.dashboard.pipeline import mount_pipeline


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


def test_mount_is_in_place_and_tolerates_stubs():
    app = FastAPI()
    assert mount_pipeline(app) is app
    stub = object()  # what tests/test_run.py hands run.py in place of make_app()
    assert mount_pipeline(stub) is stub
