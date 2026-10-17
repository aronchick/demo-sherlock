"""Local HTTP contract checks; no GPU, Docker daemon or Brev instance required."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import socket
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/brev_preflight.py"
spec = importlib.util.spec_from_file_location("brev_preflight", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def schema_example(schema):
    """Derive protocol-only arguments from today's schema, never copied fixtures."""
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {
            key: schema_example(schema["properties"][key])
            for key in schema.get("required", [])
        }
    if kind == "array":
        return [
            schema_example(schema["items"]) for _ in range(schema.get("minItems", 1))
        ]
    if kind == "string":
        return "hero"
    if kind in {"integer", "number"}:
        return schema.get("minimum", 0)
    if kind == "boolean":
        return False
    raise AssertionError(f"Unhandled runtime schema: {schema}")


def tool_choice(tools):
    function = tools[0]["function"]
    return {
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_stub",
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": json.dumps(schema_example(function["parameters"])),
                    },
                }
            ],
        },
    }


@contextmanager
def stub(*, status=200, unready=0, ids=None, reply=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            assert self.path == "/v1/models"
            with self.server.lock:
                self.server.gets += 1
                count = self.server.gets
            self.send_response(503 if count <= unready else status)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"data": [{"id": mid} for mid in (ids or ["director", "vlm"])]}
                ).encode()
            )

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with self.server.lock:
                self.server.bodies.append(body)
            choice = (
                tool_choice(body["tools"])
                if "tools" in body
                else {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "A test object."},
                }
            )
            if reply:
                choice = reply(choice, body)
            self.send_response(200)
            if body.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                delta = dict(choice["message"])
                for index, call in enumerate(delta.get("tool_calls", [])):
                    call["index"] = index
                events = [
                    {"choices": [{"delta": delta}]},
                    {
                        "choices": [
                            {"delta": {}, "finish_reason": choice["finish_reason"]}
                        ]
                    },
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                    },
                ]
                for event in events:
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [choice]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.gets, server.bodies, server.lock = 0, [], threading.Lock()
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
        print(f"Preflight stub stopped; port {port} rebound successfully")


@pytest.fixture
def crop(tmp_path):
    # A synthetic JPEG checks the transport only; it is never model evidence.
    path = tmp_path / "protocol-only.jpg"
    assert cv2.imwrite(str(path), np.full((32, 48, 3), 127, dtype=np.uint8))
    return path


def endpoint(base, model=None):
    return {"base": base, "model": model, "key": "stub-secret-not-for-output"}


def test_readiness_retries_and_selects_alias():
    with stub(unready=2) as (server, base):
        ep = endpoint(base)
        result = asyncio.run(preflight.wait_models("director", ep, 3, 0.01))
        assert result["selected"] == ep["model"] == "director"
        assert server.gets == 3


@pytest.mark.parametrize(
    "status, ids, match",
    [
        (401, None, "authentication failed"),
        (503, None, "timed out"),
        (200, ["other", "ambiguous"], "timed out"),
    ],
)
def test_readiness_failures(status, ids, match):
    with (
        stub(status=status, ids=ids) as (_, base),
        pytest.raises(ValueError, match=match),
    ):
        asyncio.run(preflight.wait_models("director", endpoint(base), 0.2, 0.01))


def test_single_model_fallback_and_explicit_missing_model():
    with stub(ids=["repository/model"]) as (_, base):
        result = asyncio.run(preflight.wait_models("director", endpoint(base), 2, 0.01))
        assert result["selected"] == "repository/model"
        with pytest.raises(ValueError, match="timed out"):
            asyncio.run(
                preflight.wait_models("director", endpoint(base, "missing"), 0.2, 0.01)
            )


def test_smoke_uses_live_prompts_schemas_and_jpeg(crop):
    from sherlock.agents import director
    from sherlock.agents.vision import Analyst
    from sherlock.tools import TOOL_SCHEMAS

    with stub() as (server, base):
        eps = {role: endpoint(base, role) for role in ("director", "vlm")}
        payloads = preflight.smoke_payloads(eps, crop)
        assert payloads["director"]["tools"] is TOOL_SCHEMAS
        assert payloads["director"]["messages"][0]["content"] == director.SYSTEM.format(
            max_calls=6 + getattr(director, "WEAPON_EXTRA_CALLS", 4)
        )
        result = asyncio.run(preflight.smoke(eps, payloads, 3))
        assert result["director"]["valid_tool_calls"] == 1
        body, vision = server.bodies
        event = json.loads(body["messages"][1]["content"])
        assert event["threat"] == "weapon" and len(event["head_bbox"]) == 4
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        contents = vision["messages"][0]["content"]
        assert contents[0]["text"] == Analyst.PROMPT.format(
            question="What is this object? Read any visible text."
        )
        assert base64.b64decode(
            contents[1]["image_url"]["url"].split(",")[1]
        ).startswith(b"\xff\xd8")


@pytest.mark.parametrize(
    "mutation",
    ["reasoning", "prose", "bad_args", "unknown_tool", "length", "empty_vlm"],
)
def test_bad_smoke_is_rejected(crop, mutation):
    def reply(choice, body):
        if "tools" in body:
            call = choice["message"]["tool_calls"][0]
            if mutation == "reasoning":
                choice["message"]["reasoning_content"] = "thinking"
            elif mutation == "prose":
                choice["message"]["tool_calls"] = []
            elif mutation == "bad_args":
                call["function"]["arguments"] = '{"unexpected": 1}'
            elif mutation == "unknown_tool":
                call["function"]["name"] = "does_not_exist"
            elif mutation == "length":
                choice["finish_reason"] = "length"
        elif mutation == "empty_vlm":
            choice["message"]["content"] = " "
        return choice

    with stub(reply=reply) as (_, base):
        eps = {role: endpoint(base, role) for role in ("director", "vlm")}
        with pytest.raises(ValueError):
            asyncio.run(preflight.smoke(eps, preflight.smoke_payloads(eps, crop), 3))


@pytest.mark.parametrize("mode", ["models", "smoke", "sweep"])
def test_real_uv_entrypoint(mode, crop, tmp_path):
    if mode == "sweep" and not preflight.BENCH.is_file():
        pytest.skip("Run combined with benchmark PR #11 to exercise the real sweep")
    with stub() as (server, base):
        env = {
            **os.environ,
            "DIRECTOR_BASE_URL": base,
            "ANALYST_BASE_URL": base,
            "DIRECTOR_MODEL": "director",
            "ANALYST_MODEL": "vlm",
            "DIRECTOR_API_KEY": "stub-secret-not-for-output",
            "ANALYST_API_KEY": "stub-secret-not-for-output",
        }
        result = subprocess.run(
            [
                "uv",
                "run",
                str(SCRIPT),
                "--mode",
                mode,
                "--image",
                str(crop),
                "--requests",
                "16",
                "--warmup",
                "1",
                "--ready-timeout",
                "3",
                "--timeout",
                "5",
                "--evidence-kind",
                "stub",
                "--output-dir",
                str(tmp_path),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "stub-secret-not-for-output" not in result.stdout + result.stderr
        manifest = json.loads(
            next(tmp_path.glob("brev_preflight_*/preflight.json")).read_text()
        )
        assert manifest["ok"] and manifest["evidence_kind"] == "stub"
        if mode == "models":
            assert not server.bodies
        else:
            assert manifest["stages"][0]["smoke"]["director"]["thinking"] is False
        if mode == "sweep":
            bench = json.loads(
                Path(manifest["stages"][0]["benchmark_json"]).read_text()
            )
            assert len(bench["runs"]) == 8
            assert all(row["summary"]["succeeded"] == 16 for row in bench["runs"])
            assert {row["concurrency"] for row in bench["runs"]} == {1, 4, 8, 16}


def test_restart_order_flags_and_invariance(monkeypatch):
    calls, state = [], {"prefix": 1}

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[:3] == ["docker", "container", "ls"]:
            return SimpleNamespace(stdout="director\nvlm\nunrelated\n")
        if argv[:3] == ["docker", "container", "inspect"]:
            flags = ["--model", argv[-1]]
            if argv[-1] == "director":
                flags += (
                    ["--enable-prefix-caching", "--prefix-caching-hash-algo", "sha256"]
                    if state["prefix"]
                    else ["--no-enable-prefix-caching"]
                )
            return SimpleNamespace(
                stdout="\n".join(
                    map(json.dumps, [preflight.VLLM_IMAGE, flags, "sha256:fixed"])
                )
            )
        if argv[0] == "bash":
            state["prefix"] = int(kwargs["env"]["PREFIX_CACHING"])
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(preflight, "command", command)
    assert preflight.restart(0) == preflight.restart(1)
    mutations = [
        argv for argv in calls if argv[0] == "bash" or argv[2] in {"stop", "rm"}
    ]
    assert (
        mutations
        == [
            ["docker", "container", "stop", "--timeout", "30", "director", "vlm"],
            ["docker", "container", "rm", "director", "vlm"],
            ["bash", str(preflight.ROOT / "brev/serve.sh")],
        ]
        * 2
    )


@pytest.mark.parametrize("drift", [False, True])
def test_ab_orders_legs_compares_and_rejects_drift(monkeypatch, crop, tmp_path, drift):
    calls = []
    monkeypatch.setattr(preflight, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(preflight, "BENCH", SCRIPT)
    monkeypatch.setattr(
        preflight,
        "endpoints",
        lambda: {
            "director": endpoint("http://127.0.0.1:8001/v1", "director"),
            "vlm": endpoint("http://127.0.0.1:8002/v1", "vlm"),
        },
    )

    def restart(prefix):
        calls.append(("restart", prefix))
        return str(prefix) if drift else "same"

    async def ready(_eps, _args):
        calls.append("ready")
        return {}

    async def smoke(*_args):
        calls.append("smoke")
        return {}

    def benchmark(_args, _eps, label, folder, baseline):
        calls.append(("bench", label, baseline))
        return folder / "vllm_bench_stub.json"

    monkeypatch.setattr(preflight, "restart", restart)
    monkeypatch.setattr(preflight, "ready", ready)
    monkeypatch.setattr(preflight, "smoke", smoke)
    monkeypatch.setattr(preflight, "benchmark", benchmark)
    result = preflight.main(
        ["--mode", "ab", "--image", str(crop), "--output-dir", str(tmp_path)]
    )
    assert result == int(drift)
    manifest = json.loads(
        next(tmp_path.glob("brev_preflight_*/preflight.json")).read_text()
    )
    baseline = Path(manifest["stages"][0]["benchmark_json"])
    assert calls[:5] == [
        ("restart", 0),
        "ready",
        "smoke",
        ("bench", "off", None),
        ("restart", 1),
    ]
    assert calls[5:] == ([] if drift else ["ready", "smoke", ("bench", "on", baseline)])
    assert manifest["ok"] is not drift


def test_benchmark_passes_compare_and_credentials_in_env(monkeypatch, crop, tmp_path):
    args = preflight.parser().parse_args(["--image", str(crop)])
    result_file = tmp_path / "vllm_bench_stub.json"
    result_file.write_text("{}")
    eps = {
        role: endpoint("http://127.0.0.1:1234/v1", role) for role in ("director", "vlm")
    }
    recorded = []
    monkeypatch.setattr(
        preflight, "command", lambda argv, **kwargs: recorded.append((argv, kwargs))
    )
    assert (
        preflight.benchmark(args, eps, "on", tmp_path, Path("off.json")) == result_file
    )
    argv, kwargs = recorded[0]
    assert argv[-2:] == ["--compare", "off.json"]
    assert "--metrics" in argv and argv[argv.index("--prefix-caching") + 1] == "on"
    assert kwargs["env"]["DIRECTOR_MODEL"] == "director"
    assert kwargs["env"]["DIRECTOR_API_KEY"] == "stub-secret-not-for-output"
    assert not any("stub-secret" in arg for arg in argv)


def test_restart_refuses_unrelated_image_before_any_stop(monkeypatch):
    calls = []

    def command(argv, **_kwargs):
        calls.append(argv)
        if argv[2] == "ls":
            return SimpleNamespace(stdout="director\nvlm\n")
        return SimpleNamespace(stdout='"unrelated/image"\n[]\n"sha256:fixed"')

    monkeypatch.setattr(preflight, "command", command)
    with pytest.raises(ValueError, match="Refusing to restart"):
        preflight.restart(0)
    assert all(argv[2] in {"ls", "inspect"} for argv in calls)


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--enable-prefix-caching", "--no-enable-prefix-caching"],
        ["--enable-prefix-caching", "--speculative-config", "{}"],
    ],
)
def test_restart_rejects_incorrect_effective_flags(monkeypatch, flags):
    monkeypatch.setattr(
        preflight, "command", lambda *_args, **_kwargs: SimpleNamespace(stdout="")
    )
    monkeypatch.setattr(
        preflight,
        "inspect_container",
        lambda _name: {"argv": list(flags), "image_id": "sha256:fixed"},
    )
    with pytest.raises(ValueError, match="startup flags"):
        preflight.restart(1)


def test_ab_real_http_and_benchmark_comparison(monkeypatch, tmp_path):
    if not preflight.BENCH.is_file():
        pytest.skip("Run combined with benchmark PR #11 for the real A/B protocol test")
    restarts = []
    original_ready = preflight.ready
    monkeypatch.setattr(preflight, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(
        preflight,
        "endpoints",
        lambda: {
            "director": endpoint("http://127.0.0.1:8001/v1", "director"),
            "vlm": endpoint("http://127.0.0.1:8002/v1", "vlm"),
        },
    )

    def restart(prefix):
        restarts.append(prefix)
        return "fake-docker-unchanged-config"

    with stub() as (_, base):

        async def ready(eps, args):
            # Redirect checked local ports to this test's ephemeral HTTP server.
            # Only Docker is faked; smoke, subprocess benchmark and comparison run.
            for ep in eps.values():
                ep["base"] = base
            return await original_ready(eps, args)

        monkeypatch.setattr(preflight, "restart", restart)
        monkeypatch.setattr(preflight, "ready", ready)
        result = preflight.main(
            [
                "--mode",
                "ab",
                "--requests",
                "16",
                "--warmup",
                "1",
                "--ready-timeout",
                "3",
                "--timeout",
                "5",
                "--evidence-kind",
                "stub",
                "--output-dir",
                str(tmp_path),
            ]
        )
    assert result == 0 and restarts == [0, 1]
    manifest = json.loads(
        next(tmp_path.glob("brev_preflight_*/preflight.json")).read_text()
    )
    assert manifest["ok"]
    baseline, candidate = [
        json.loads(Path(stage["benchmark_json"]).read_text())
        for stage in manifest["stages"]
    ]
    assert baseline["prefix_caching"] == "off" and candidate["prefix_caching"] == "on"
    assert candidate["evidence_kind"] == "stub"
    assert len(candidate["comparison"]["results"]) == 8
    assert all(
        row["summary"]["failed"] == 0
        for leg in (baseline, candidate)
        for row in leg["runs"]
    )


def test_mac_commands_are_copyable_and_secret_free(capsys):
    args = preflight.parser().parse_args(["--instance", "sherlock-gpu"])
    eps = {role: endpoint("http://host/v1", role) for role in ("director", "vlm")}
    preflight.mac_instructions(args, eps)
    output = capsys.readouterr().out
    assert "stub-secret" not in output
    assert "\n+" not in output
    assert "VERIFIER_BASE_URL=http://127.0.0.1:8002/v1" in output
    assert "\nVERIFIER_MARGIN=-1.0\n" in output
    assert "brev port-forward sherlock-gpu --port 8001:8001" in output
    assert "brev port-forward sherlock-gpu --port 8002:8002" in output
    assert "-L 127.0.0.1:8002:127.0.0.1:8002 sherlock-gpu" in output
