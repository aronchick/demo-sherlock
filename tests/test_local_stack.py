"""scripts/local.sh, the Plan B launcher behind `just local-up` / `just local-down`. These never
start a model: every test runs on a scratch runtime directory and ports nobody listens on, so
a real local stack (or anyone else's server) on 11436 and 8002 is never touched."""
import os
import pathlib
import shutil
import socket
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "local.sh"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def env(tmp_path):
    e = {k: v for k, v in os.environ.items() if k != "VERIFIER_MARGIN"}
    e.update(LOCAL_RUNTIME=str(tmp_path / "runtime"), LOCAL_DIRECTOR_PORT=str(_free_port()),
             LOCAL_VLM_PORT=str(_free_port()))
    return e


def _run(env, *args, timeout=30):
    return subprocess.run([str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=timeout,
                          check=False)


def test_no_command_prints_usage():
    r = subprocess.run([str(SCRIPT)], capture_output=True, text=True, timeout=10, check=False)
    assert r.returncode == 2
    assert "scripts/local.sh down" in r.stdout and "Apple silicon only" in r.stdout


def test_env_points_every_model_at_loopback_with_the_demo_margin(env):
    r = _run(env, "env")
    assert r.returncode == 0, r.stderr
    exports = dict(line.removeprefix("export ").split("=", 1) for line in r.stdout.splitlines())
    d, v = env["LOCAL_DIRECTOR_PORT"], env["LOCAL_VLM_PORT"]
    assert exports["DIRECTOR_BASE_URL"] == f"http://127.0.0.1:{d}/v1"
    assert exports["ANALYST_BASE_URL"] == exports["VERIFIER_BASE_URL"] == f"http://127.0.0.1:{v}/v1"
    assert exports["DIRECTOR_MODEL"] == "nemotron-3.5-lightning:30b-mlx"
    # One id everywhere, or mlx-vlm loads a second copy of the model.
    assert exports["ANALYST_MODEL"] == exports["VERIFIER_MODEL"] == "HuggingFaceTB/SmolVLM-500M-Instruct"
    assert exports["VERIFIER_MARGIN"] == "-1.0"
    assert _run({**env, "VERIFIER_MARGIN": "0.5"}, "env").stdout.splitlines()[-1] == "export VERIFIER_MARGIN=0.5"


def test_up_refuses_anything_but_apple_silicon(env, tmp_path):
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "uname").write_text('#!/bin/sh\n[ "$1" = -m ] && echo x86_64 || echo Linux\n')
    (shim / "uname").chmod(0o755)
    r = _run({**env, "PATH": f"{shim}:{env['PATH']}"}, "up")
    assert r.returncode == 1
    assert "Apple silicon" in r.stderr and "Brev" in r.stderr
    assert not (tmp_path / "runtime").exists()  # refused before it started or wrote anything


def test_live_refuses_when_the_models_are_not_answering(env):
    r = _run(env, "live", "--synthetic")
    assert r.returncode == 1
    assert "just local-up" in r.stderr


def test_down_with_nothing_started_is_a_clean_no_op(env):
    r = _run(env, "down")
    assert r.returncode == 0, r.stderr
    assert f"ports {env['LOCAL_DIRECTOR_PORT']} and {env['LOCAL_VLM_PORT']} are free" in r.stdout
    assert _run(env, "status").stdout.splitlines() == ["director: not running here", "vlm: not running here"]


def test_down_never_kills_a_recycled_pid(env):
    runtime = pathlib.Path(env["LOCAL_RUNTIME"])
    runtime.mkdir()
    bystander = subprocess.Popen(["sleep", "30"])
    try:
        (runtime / "director.pid").write_text(f"{bystander.pid}\n")
        r = _run(env, "down")
        assert r.returncode == 0, r.stderr
        assert bystander.poll() is None  # alive: it is not `ollama serve`, so it is not ours
        assert not (runtime / "director.pid").exists()
    finally:
        bystander.kill()
        bystander.wait()


@pytest.mark.skipif(shutil.which("lsof") is None, reason="port checks use lsof")
def test_down_fails_while_a_port_is_still_held(env):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        held = s.getsockname()[1]
        r = _run({**env, "LOCAL_VLM_PORT": str(held)}, "down")
    assert r.returncode == 1
    assert f"still in use: {held}" in r.stderr


def test_just_recipes_drive_the_script():
    justfile = (ROOT / "justfile").read_text()
    for recipe, command in (("local-up:", "up"), ("local-live *args:", "live"), ("local-status:", "status"),
                            ("local-down:", "down")):
        assert f"{recipe}\n    scripts/local.sh {command}" in justfile
