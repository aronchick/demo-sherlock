"""Exercise real HTTP streaming and the CLI without a model or GPU."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bench_vllm.py"
spec = importlib.util.spec_from_file_location("bench_vllm", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def example_arguments(name):
    return bench.tool_arguments(
        name,
        {
            "camera": "hero",
            "bbox": [1, 2, 3, 4],
            "subject": "weapon",
            "question": "What is this object?",
            "claim": "A kitchen knife.",
            "confidence": 0.9,
            "action": "alert",
            "evidence": ["weapon_crop", "face_crop"],
            "pan": 0,
            "tilt": 0,
            "zoom": 2,
        },
    )


@contextmanager
def stub(mode="normal"):
    release_response = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            assert self.path == "/metrics"
            if mode == "no_metrics":
                self.send_response(404)
                self.end_headers()
                return
            with self.server.lock:
                count = self.server.count
            body = (
                "\n".join(
                    f'{name}{{model_name="{model}",engine="0"}} {value}'
                    for model in ("director", "vlm")
                    for name, value in (
                        ("vllm:prefix_cache_queries_total", count * 100),
                        ("vllm:prefix_cache_hits_total", count * 50),
                        ("vllm:kv_cache_usage_perc", 0.25),
                    )
                )
                + "\n"
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with self.server.lock:
                self.server.count += 1
                request_number = self.server.count
                self.server.bodies.append(body)
            if mode == "http_error":
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def event(data):
                self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
                self.wfile.flush()

            try:
                event({"choices": [{"delta": {"role": "assistant"}}]})
                if mode == "timeout":
                    # Only teardown releases this stream, so scheduler delays
                    # cannot let a late client read a completed response.
                    release_response.wait()
                    return
                time.sleep(0.002)
                if body["model"] == "director" and mode != "prose":
                    # Exercise every imported tool, including new tools, over
                    # real fragmented SSE. This stub checks structure, not policy.
                    name = body["tools"][request_number % len(body["tools"])][
                        "function"
                    ]["name"]
                    arguments = (
                        json.dumps(example_arguments(name))
                        if mode != "bad_tool"
                        else '{"unknown_argument":5}'
                    )
                    event(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_1",
                                                "type": "function",
                                                "function": {
                                                    "name": name,
                                                    "arguments": arguments[:9],
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    )
                    time.sleep(0.002)
                    event(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {
                                                    "arguments": arguments[9:]
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    )
                    finish = "tool_calls"
                else:
                    for content in ("A white model ", "spacecraft with dark windows."):
                        event({"choices": [{"delta": {"content": content}}]})
                    finish = "stop"
                if mode == "truncated":
                    return
                event(
                    {
                        "choices": [
                            {
                                "delta": {},
                                "finish_reason": (
                                    "length" if mode == "length" else finish
                                ),
                            }
                        ]
                    }
                )
                if mode != "no_usage":
                    event(
                        {
                            "choices": [],
                            "usage": {"prompt_tokens": 150, "completion_tokens": 12},
                        }
                    )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False  # server_close joins the released handlers.
    server.count, server.bodies, server.lock = 0, [], threading.Lock()
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{port}/v1"
    finally:
        release_response.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
        print(f"Stub stopped; port {port} is free")


def run_cli(base, output, *flags, expected=0):
    env = {
        **os.environ,
        "DIRECTOR_BASE_URL": base,
        "ANALYST_BASE_URL": base,
        "DIRECTOR_MODEL": "director",
        "ANALYST_MODEL": "vlm",
        "DIRECTOR_API_KEY": "stub-only",
        "ANALYST_API_KEY": "stub-only",
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--requests",
            "16",
            "--warmup",
            "1",
            "--metrics",
            "--evidence-kind",
            "stub",
            "--output-dir",
            str(output),
            *flags,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == expected, proc.stdout + proc.stderr
    paths = sorted(output.glob("vllm_bench_*.json"))
    assert paths
    return paths[-1], json.loads(paths[-1].read_text())


def test_cli_full_sweep_and_ab(tmp_path):
    with stub() as (server, base):
        off_path, off = run_cli(base, tmp_path / "off", "--prefix-caching", "off")
        on_path, on = run_cli(
            base, tmp_path / "on", "--prefix-caching", "on", "--compare", str(off_path)
        )
        assert len(on["comparison"]["results"]) == 8
        assert len(off["runs"]) == 8
        assert on["evidence_kind"] == "stub"
        assert "stub-only" not in on_path.read_text()
        assert base not in on_path.read_text()
        assert "**stub**" in on_path.with_suffix(".md").read_text()
        for row in on["runs"]:
            summary = row["summary"]
            assert summary["succeeded"] == 16 and summary["failed"] == 0
            assert summary["output_tokens"] == 192
            assert summary["output_tokens_per_s"] > 0
            assert summary["e2e_ms"]["p50"] >= summary["ttft_ms"]["p50"] > 0
            assert row["metrics"]["prefix_cache_hit_rate"] == 0.5
            if row["workload"] == "director":
                assert summary["tool_call_validity_rate"] == 1
        assert server.count == 2 * 8 * 17
        director = next(b for b in server.bodies if b["model"] == "director")
        threat = json.loads(director["messages"][1]["content"])["threat"]
        budget = 6 + (
            getattr(bench.director, "WEAPON_EXTRA_CALLS", 4)
            if threat == "weapon"
            else 0
        )
        assert director["messages"][0]["content"] == bench.SYSTEM.format(
            max_calls=budget
        )
        assert director["tools"] == bench.TOOL_SCHEMAS
        assert director["chat_template_kwargs"] == {"enable_thinking": False}
        assert (
            max(len(b["messages"]) for b in server.bodies if b["model"] == "director")
            >= 6
        )
        vision = next(b for b in server.bodies if b["model"] == "vlm")
        data = vision["messages"][0]["content"][1]["image_url"]["url"]
        image = bench.cv2.imdecode(
            bench.np.frombuffer(
                bench.base64.b64decode(data.split(",")[1]), bench.np.uint8
            ),
            bench.cv2.IMREAD_COLOR,
        )
        assert image.shape == (240, 110, 3)


def test_director_routing_and_history_match_runtime_schemas():
    seen = set()
    for index in range(72):
        body = bench.payload("director", index, "director", "unused", 6)
        event = json.loads(body["messages"][1]["content"])
        _, plan = bench.director_case(index)
        report_name, report_args, _result = plan[-1]
        assert report_name == "report"
        assert (
            report_args["action"]
            == {"weapon": "alert", "benign": "ignore", "unknown": "log"}[
                event["threat"]
            ]
        )
        if "evidence" in bench.PARAMETERS["report"]["properties"]:
            evidence = (
                []
                if event["threat"] == "benign"
                else ["weapon_crop" if event["threat"] == "weapon" else "object_crop"]
            )
            if event["head_bbox"]:
                evidence.append("face_crop")
            assert report_args["evidence"] == evidence
        assert event["threat"] in {"weapon", "benign", "unknown"}
        assert "head_bbox" in event
        budget = 6 + (
            getattr(bench.director, "WEAPON_EXTRA_CALLS", 4)
            if event["threat"] == "weapon"
            else 0
        )
        assert body["messages"][0]["content"] == bench.SYSTEM.format(max_calls=budget)
        seen.add(
            (event["threat"], event["head_bbox"] is not None, len(body["messages"]))
        )
        calls = []
        for message in body["messages"][2::2]:
            call = message["tool_calls"][0]
            assert bench.valid_call({**call, **call["function"]})
            calls.append(
                (call["function"]["name"], json.loads(call["function"]["arguments"]))
            )
        assert all(name != "report" for name, _args in calls)
        if event["threat"] == "benign":
            assert not calls
        else:
            if calls:
                assert calls[0] == (
                    "look_at",
                    {"camera": "hero", "bbox": event["bbox"]},
                )
            if (
                len(calls) >= 2
                and "subject" in bench.PARAMETERS["snapshot"]["properties"]
            ):
                assert calls[1][1]["subject"] == (
                    "weapon" if event["threat"] == "weapon" else "object"
                )
            if len(calls) >= 4:
                assert calls[2] == (
                    "look_at",
                    {"camera": "hero", "bbox": event["head_bbox"]},
                )
                assert calls[3] == ("snapshot", {"camera": "hero", "subject": "face"})
                assert json.loads(body["messages"][9]["content"]) == {
                    "evidence_id": "face_crop",
                    "stored": True,
                    "analyzed": False,
                }
            if len(calls) == 5:
                assert calls[4][0] == "verify"
    assert {threat for threat, _head, _length in seen} == {
        "weapon",
        "benign",
        "unknown",
    }
    if "verify" in bench.PARAMETERS:
        assert ("weapon", True, 12) in seen  # complete chain before report
        assert ("weapon", False, 8) in seen  # no face steps without a head box
        assert ("unknown", False, 8) in seen


@pytest.mark.parametrize("name", bench.PARAMETERS)
def test_tool_requirements_and_enums_come_from_runtime_schema(name):
    args = example_arguments(name)

    def valid(values):
        return bench.valid_call(
            {
                "id": "x",
                "type": "function",
                "name": name,
                "arguments": json.dumps(values),
            }
        )

    assert valid(args)
    schema = bench.PARAMETERS[name]
    for required in schema.get("required", []):
        assert not valid({key: value for key, value in args.items() if key != required})
    for field, properties in schema["properties"].items():
        if "enum" in properties:
            for choice in properties["enum"]:
                assert valid({**args, field: choice})
            assert not valid({**args, field: "not_a_runtime_enum_value"})


def test_vlm_prompts_ask_only_about_the_object():
    analyst = bench.payload("vlm", 0, "vlm", "image-data", 6)
    verifier = bench.payload("vlm", 1, "vlm", "image-data", 6)
    assert analyst["messages"][0]["content"][0]["text"] == bench.Analyst.PROMPT.format(
        question="What is this object? Read any visible text."
    )
    assert verifier["messages"][0]["content"][0][
        "text"
    ] == bench.Verifier.PROMPT.format(
        claim="a white model spacecraft with dark windows"
    )


@pytest.mark.parametrize(
    "mode,error",
    [
        ("http_error", "http_503"),
        ("truncated", "incomplete_stream"),
        ("length", "length"),
        ("timeout", "TimeoutError"),
    ],
)
def test_failures_are_counted_and_written(tmp_path, mode, error):
    with stub(mode) as (_server, base):
        _, result = run_cli(
            base,
            tmp_path,
            "--concurrency",
            "1",
            "--requests",
            "1",
            "--workloads",
            "director",
            "--timeout",
            ".2" if mode == "timeout" else "5",
            expected=1,
        )
    summary = result["runs"][0]["summary"]
    assert summary["failed"] == 1
    assert summary["errors"] == {error: 1}
    assert summary["requests_per_s"] == 0
    assert summary["ttft_ms"]["p50"] is None
    assert summary["output_tokens_per_s"] is None
    assert summary["tool_call_validity_rate"] == 0


@pytest.mark.parametrize("mode", ["no_usage", "bad_tool", "prose", "no_metrics"])
def test_missing_usage_and_invalid_tool_do_not_become_good_evidence(tmp_path, mode):
    with stub(mode) as (_server, base):
        _, result = run_cli(
            base,
            tmp_path,
            "--concurrency",
            "1",
            "--requests",
            "1",
            "--workloads",
            "director",
        )
    summary = result["runs"][0]["summary"]
    assert summary["succeeded"] == 1
    if mode == "no_usage":
        assert summary["output_tokens_per_s"] is None
        assert summary["usage_coverage"] == 0
    elif mode in ("bad_tool", "prose"):
        assert summary["tool_call_validity_rate"] == 0
    else:
        metrics = result["runs"][0]["metrics"]
        assert metrics["prefix_cache_hit_rate"] is None
        assert metrics["before"]["available"] is False


def test_nonfinite_tool_arguments_are_not_json():
    assert not bench.valid_call(
        {
            "id": "x",
            "type": "function",
            "name": "report",
            "arguments": '{"claim":"x","confidence":NaN,"action":"log"}',
        }
    )


@pytest.mark.parametrize(
    "name,args,valid",
    [
        ("look_at", {"camera": "hero", "bbox": [1, 2, 3, 4]}, True),
        ("look_at", {"camera": "hero", "bbox": [1, 2, 3]}, False),
        ("report", {"claim": "person", "confidence": 1.2, "action": "alert"}, False),
        ("report", {"claim": "person", "confidence": 0.8, "action": "shoot"}, False),
        ("look_at", {"camera": "hero", "bbox": [1, 2, 3, 4], "typo": 1}, False),
        ("invented_tool", {}, False),
        ("list_cameras", {}, True),
    ],
)
def test_tool_validity(name, args, valid):
    assert (
        bench.valid_call(
            {"id": "x", "type": "function", "name": name, "arguments": json.dumps(args)}
        )
        is valid
    )


def test_metrics_filter_models_and_detect_reset():
    raw = (
        'vllm:prefix_cache_queries_total{model_name="director"} 10\n'
        'vllm:prefix_cache_queries_total{model_name="unrelated"} 999\n'
        'vllm:prefix_cache_hits_total{model_name="director"} 5\n'
        'vllm:kv_cache_usage_perc{model_name="director",engine="0"} 0.2\n'
        'vllm:kv_cache_usage_perc{model_name="director",engine="1"} 0.3\n'
    )
    snapshot = bench.metric_values(raw, "director")
    values = snapshot["values"]
    assert snapshot["model_match"] == "exact"
    assert values["vllm:prefix_cache_queries_total"] == 10
    assert values["vllm:kv_cache_usage_perc"] == 0.3
    before = {"values": values}
    reset = {
        "values": {
            "vllm:prefix_cache_queries_total": 1,
            "vllm:prefix_cache_hits_total": 0,
        }
    }
    assert bench.metrics_delta(before, reset)["prefix_cache_hit_rate"] is None
    assert bench.metrics_delta(before, before)["prefix_cache_hit_rate"] is None
    assert bench.metrics_delta({"values": {}}, before)["prefix_cache_hit_rate"] is None


def test_metrics_match_single_model_served_under_an_alias():
    raw = 'vllm:prefix_cache_hits_total{model_name="org/model-repo"} 12\n'
    snapshot = bench.metric_values(raw, "director")
    assert snapshot["values"]["vllm:prefix_cache_hits_total"] == 12
    assert snapshot["matched_model_name"] == "org/model-repo"
    assert snapshot["model_match"] == "single_model_alias"
    ambiguous = raw + 'vllm:prefix_cache_hits_total{model_name="another/model"} 50\n'
    assert bench.metric_values(ambiguous, "director")["values"] == {}


def test_summary_denominators_and_ab_guards():
    good = {
        "ok": True,
        "error": None,
        "ttft_ms": 10,
        "e2e_ms": 50,
        "output_tokens": 20,
        "tool_calls": 1,
        "valid_tool_calls": 1,
        "tool_response_valid": True,
    }
    bad = {
        **good,
        "ok": False,
        "error": "timeout",
        "tool_response_valid": False,
        "tool_calls": 0,
        "valid_tool_calls": 0,
    }
    summary = bench.summarize([good, bad], 2, "director")
    assert summary["requests_per_s"] == 0.5
    assert summary["output_tokens_per_s"] == 10
    assert summary["tool_call_validity_rate"] == 0.5
    assert summary["individual_tool_call_validity_rate"] == 1
    base = {
        "schema_version": 1,
        "created_at": "now",
        "prefix_caching": "off",
        "evidence_kind": "stub",
        "workload_fingerprint": "x",
        "settings": {},
        "environment": "H100",
        "runs": [],
    }
    candidate = {**base, "prefix_caching": "on"}
    comparison = bench.compare_runs(base, candidate)
    assert comparison["results"] == []
    assert comparison["baseline_workload_fingerprint"] == "x"
    assert comparison["candidate_workload_fingerprint"] == "x"
    for field in (
        "evidence_kind",
        "workload_fingerprint",
        "environment",
        "settings",
        "prefix_caching",
    ):
        altered = copy.deepcopy(candidate)
        altered[field] = "different"
        with pytest.raises(ValueError):
            bench.compare_runs(base, altered)


def test_endpoint_paths_and_credentials():
    assert (
        bench.endpoint("http://localhost:8001", "chat/completions")
        == "http://localhost:8001/v1/chat/completions"
    )
    assert (
        bench.endpoint("https://host/prefix/v1/", "metrics")
        == "https://host/prefix/metrics"
    )
    for base in (
        "http://user:pass@host/v1",
        "https://host/v1?token=secret",
        "file:///tmp/file",
    ):
        with pytest.raises(ValueError):
            bench.endpoint(base, "metrics")


def test_serving_metadata_probes_endpoint(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            body = {"data": [{"owned_by": "vllm"}]} if self.path == "/v1/models" else {"version": "0.28.0"}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    for name in ("MODEL_SERVER", "DIRECTOR_MODEL_SERVER", "ANALYST_MODEL_SERVER", "BREV_HOST"):
        monkeypatch.delenv(name, raising=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DIRECTOR_BASE_URL", base)
        monkeypatch.setenv("ANALYST_BASE_URL", base + "/v1")
        bench.model_servers.clear_cache()
        args = bench.parser().parse_args([])
        metadata = bench.serving_metadata(args)
        assert metadata["model_server"] == {
            "name": "vLLM", "version": "0.28.0", "host": "laptop",
            "label": "vLLM", "reachable": True, "source": "detected",
        }
        assert all(value == metadata["model_server"] for value in metadata["model_servers"].values())
        args.evidence_kind = "stub"
        monkeypatch.setattr(bench.model_servers, "server", lambda *_: pytest.fail("stub probed"))
        assert bench.serving_metadata(args) == {"model_server": None, "model_servers": {}}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        bench.model_servers.clear_cache()


def test_imported_brev_metadata_is_declared():
    files = list(SCRIPT.parents[1].glob("results/brev_preflight_*/**/vllm_bench_*.json"))
    assert files
    for path in files:
        metadata = json.loads(path.read_text())["model_server"]
        assert metadata["name"] == "vLLM"
        assert metadata["version"] == "0.28.0"
        assert metadata["host"] == "sherlock-h100"
        assert metadata["source"] == "declared"
