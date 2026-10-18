"""The pages name the program serving each model only from what that server answers:
vLLM on Brev, or the local fallback (Ollama Director, mlx-vlm VLM), never assumed."""
import json
import pathlib
import re
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sherlock.config import Config, Endpoint
from sherlock.dashboard import model_servers
from sherlock.dashboard.pipeline import mount_pipeline
from sherlock.dashboard.server import STATIC, make_app

# Response shapes copied from each server: vLLM's ModelCard (owned_by "vllm") and
# /version, Ollama 0.40.5, mlx-vlm 0.7.8's ModelsResponse and FastAPI title.
VLLM = {"/v1/models": {"object": "list", "data": [{"id": "RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8",
                                                   "object": "model", "owned_by": "vllm"}]},
        "/version": {"version": "0.28.0"}}
OLLAMA = {"/v1/models": {"object": "list", "data": [{"id": "nemotron-3.5-lightning:30b-mlx", "object": "model",
                                                     "created": 1, "owned_by": "library"}]},
          "/api/version": {"version": "0.40.5"}}
MLX_VLM = {"/v1/models": {"object": "list", "data": [{"id": "HuggingFaceTB/SmolVLM-500M-Instruct",
                                                      "object": "model", "created": 1, "loaded": True}]},
           "/openapi.json": {"openapi": "3.1.0", "info": {"title": "MLX-VLM Inference API", "version": "0.7.8"}}}
OTHER = {"/v1/models": {"object": "list", "data": [{"id": "some-model", "object": "model"}]}}


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    for key in ("MODEL_SERVER", "DIRECTOR_MODEL_SERVER", "ANALYST_MODEL_SERVER", "VERIFIER_MODEL_SERVER", "BREV_HOST"):
        monkeypatch.delenv(key, raising=False)
    model_servers.clear_cache()
    yield
    model_servers.clear_cache()


@contextmanager
def fake_server(routes: dict) -> Iterator[str]:
    """A local HTTP server answering GETs with canned JSON; stopped on exit."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = routes.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def closed_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


@pytest.mark.parametrize("routes, expected", [
    (VLLM, {"name": "vLLM", "version": "0.28.0", "reachable": True}),
    (OLLAMA, {"name": "Ollama", "version": "0.40.5", "reachable": True}),
    (MLX_VLM, {"name": "mlx-vlm", "version": "0.7.8", "reachable": True}),
    (OTHER, {"name": None, "version": None, "reachable": True}),
])
def test_each_server_is_named_by_what_it_answers(routes, expected):
    with fake_server(routes) as root:
        assert model_servers.identify(Endpoint(root + "/v1", "m")) == expected


def test_an_unreachable_endpoint_is_never_named():
    assert model_servers.identify(Endpoint(closed_port_url(), "m")) == {
        "name": None, "version": None, "reachable": False}


def test_local_fallback_mode_names_ollama_and_mlx_vlm_not_vllm():
    with fake_server(OLLAMA) as ollama, fake_server(MLX_VLM) as mlx:
        vlm = Endpoint(mlx + "/v1", "HuggingFaceTB/SmolVLM-500M-Instruct")
        cfg = Config(mock=False, director=Endpoint(ollama + "/v1", "nemotron-3.5-lightning:30b-mlx"),
                     analyst=vlm, verifier=Endpoint(vlm.base_url, vlm.model))
        r = TestClient(make_app(SimpleNamespace(cfg=cfg, cameras={}))).get("/mode")
    servers = r.json()["servers"]
    assert servers["director"] == {"name": "Ollama", "version": "0.40.5", "reachable": True,
                                   "source": "detected", "host": "laptop", "label": "Ollama"}
    assert servers["analyst"]["label"] == servers["verifier"]["label"] == "mlx-vlm"
    assert servers["endpoints"] == 2 and servers["vlm_shared"] is True and servers["local"] is True
    assert "vLLM" not in r.text


def test_a_mock_run_asks_no_model_server(monkeypatch):
    def refuse(ep):
        raise AssertionError(f"probed {ep.base_url} in a mock run")

    monkeypatch.setattr(model_servers, "_probe", refuse)
    r = TestClient(make_app(SimpleNamespace(cfg=Config(mock=True), cameras={}))).get("/mode")
    assert r.json()["servers"] is None


def test_the_pages_carry_what_each_server_said(monkeypatch):
    with fake_server(VLLM) as root:
        monkeypatch.delenv("SHERLOCK_MOCK", raising=False)
        for role in ("DIRECTOR", "ANALYST", "VERIFIER"):
            monkeypatch.setenv(f"{role}_BASE_URL", root + "/v1")
        client = TestClient(mount_pipeline(make_app(SimpleNamespace(cfg=Config(), cameras={}))))
        for path in ("/", "/topology"):
            cfg = json.loads(re.search(r"const CFG = (\{.*?\});\n", client.get(path).text).group(1))
            assert cfg["servers"]["director"]["label"] == "vLLM" and cfg["servers"]["director"]["version"] == "0.28.0"
            assert cfg["servers"]["endpoints"] == 1


def test_an_unknown_server_is_shown_by_its_host_never_as_vllm():
    with fake_server(OTHER) as root:
        assert model_servers.server("director", Endpoint(root + "/v1", "m"))["label"] == "laptop"
    gone = model_servers.server("analyst", Endpoint(closed_port_url(), "m"))
    assert gone["label"] == "laptop (not answering)" and gone["name"] is None


def test_model_server_env_declares_the_runtime_without_asking(monkeypatch):
    def refuse(ep):
        raise AssertionError("a declared server must not be probed")

    monkeypatch.setattr(model_servers, "_probe", refuse)
    monkeypatch.setenv("MODEL_SERVER", "vLLM")
    monkeypatch.setenv("ANALYST_MODEL_SERVER", "mlx-vlm")  # per role wins over MODEL_SERVER
    ep = Endpoint("http://127.0.0.1:9/v1", "m")
    assert model_servers.server("director", ep) == {"name": "vLLM", "version": None, "reachable": None,
                                                     "source": "declared", "host": "laptop", "label": "vLLM"}
    assert model_servers.server("analyst", ep)["label"] == "mlx-vlm"


def test_local_means_every_endpoint_is_on_this_laptop(monkeypatch):
    monkeypatch.setenv("MODEL_SERVER", "vLLM")  # no probes: only where the endpoints point matters
    local = Endpoint("http://127.0.0.1:8001/v1", "m")
    assert model_servers.describe(Config(mock=False, director=local, analyst=local, verifier=local))["local"] is True
    remote = Endpoint("https://gpu.brev.example/v1", "m")
    assert model_servers.describe(Config(mock=False, director=remote, analyst=local, verifier=local))["local"] is False
    monkeypatch.setenv("BREV_HOST", "brev-box")  # a tunnel: localhost is really the GPU box
    tunnelled = model_servers.describe(Config(mock=False, director=local, analyst=local, verifier=local))
    assert tunnelled["local"] is False and tunnelled["director"]["host"] == "brev-box"


@pytest.mark.parametrize("page", ["app.js", "topology.html", "pipeline.html"])
def test_no_page_hard_codes_a_runtime_version_or_port(page):
    text = (pathlib.Path(STATIC) / page).read_text()
    for claim in ("vLLM", "v0.28.0", ":8001", ":8002", "SmolVLM 2.2B"):
        assert claim not in text, f"{page} hard-codes {claim!r}"
