"""Judge entry points exercise real HTTP/CLI plumbing without models or Docker."""

import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sherlock.dashboard.numbers import director_result, read_numbers

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vllm_stub", ROOT / "scripts/vllm_stub.py")
stub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stub)


@pytest.mark.parametrize("script,extra,requests", [
    ("bench_vllm.py", [], 200),
    ("brev_preflight.py", ["--mode", "ab"], 16),
])
def test_judge_cli_full_flow(tmp_path, script, extra, requests):
    # Any accidental restart fails loudly; inherited endpoint settings must lose.
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text("#!/bin/sh\nexit 99\n")
    docker.chmod(0o755)
    env = {**os.environ, "PATH": f"{binary}{os.pathsep}{os.environ['PATH']}"}
    for prefix in ("DIRECTOR", "ANALYST", "VERIFIER"):
        env[f"{prefix}_BASE_URL"] = "http://127.0.0.1:9/v1"
        env[f"{prefix}_MODEL"] = "unavailable-live-model"
        env[f"{prefix}_API_KEY"] = "must-not-appear"
    output = tmp_path / "output"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--stub", "--evidence-kind", "endpoint",
         "--requests", str(requests), "--warmup", "1", "--output-dir", str(output), *extra],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "must-not-appear" not in proc.stdout + proc.stderr
    assert "Paste these lines into the Mac .env" not in proc.stdout
    port = int(re.search(r"Stub stopped; port (\d+) is free", proc.stdout)[1])
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    files = sorted(output.rglob("vllm_bench_*.json"))
    assert len(files) == (2 if extra else 1)
    for path in files:
        result = json.loads(path.read_text())
        assert result["evidence_kind"] == "stub" and result["execution_mode"] == "local_stub"
        assert result["settings"]["models"] == {"director": "director", "vlm": "vlm"}
        assert len(result["runs"]) == 8
        assert all(row["summary"]["succeeded"] == requests for row in result["runs"])
        assert director_result(result) is None
        assert f"Requests per cell: {requests}." in path.with_suffix(".md").read_text()
        if result["prefix_caching"] == "on":
            assert len(result["comparison"]["results"]) == 8
    if extra:
        report = json.loads(next(output.rglob("preflight.json")).read_text())
        assert report["ok"] and report["evidence_kind"] == "stub"
        assert all("simulated" in stage["restart"] for stage in report["stages"])
        assert all(stage["smoke"]["director"]["thinking"] is False for stage in report["stages"])


def test_stub_defaults_override_and_restore_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DIRECTOR_BASE_URL", "http://127.0.0.1:9/v1")
    args = SimpleNamespace(output_dir=None, evidence_kind="endpoint", environment="real")

    def fail(args):
        assert args.output_dir == tmp_path / ".runtime/vllm-stub"
        assert args.evidence_kind == "stub" and args.environment == "local-stub-no-model"
        assert os.environ["DIRECTOR_MODEL"] == "director"
        assert os.environ["DIRECTOR_BASE_URL"] != "http://127.0.0.1:9/v1"
        assert read_numbers(tmp_path)["director"] is None
        raise RuntimeError("intentional failure")

    with pytest.raises(RuntimeError, match="intentional failure"):
        stub.run_local(args, fail, tmp_path)
    assert os.environ["DIRECTOR_BASE_URL"] == "http://127.0.0.1:9/v1"


@pytest.mark.parametrize("linked", [False, True])
def test_stub_rejects_dashboard_results_directory(tmp_path, linked):
    results = tmp_path / "results"
    results.mkdir()
    output = results / "stub"
    if linked:
        output = tmp_path / "alias"
        output.symlink_to(results, target_is_directory=True)
    args = SimpleNamespace(output_dir=output)
    with pytest.raises(ValueError, match="outside results"):
        stub.run_local(args, lambda _: pytest.fail("must not start a run"), tmp_path)
