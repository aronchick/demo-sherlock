# Sherlock. Needs `uv` and `just`; `uv run` syncs the environment on first use.
set shell := ["bash", "-euo", "pipefail", "-c"]

# Dashboard port. Set DASHBOARD_PORT in the shell to change it.
port := env("DASHBOARD_PORT", "8000")

# Cameras for `just up`. Shell only: a live .env never leaks into the sandbox demo,
# because python-dotenv does not override variables that are already set.
cameras := env("CAMERAS", "hero:fake:synthetic,scout1:fake:synthetic")

# PID file and log for `up` / `down`; gitignored.
runtime := ".runtime"

_default:
    @"{{ just_executable() }}" --justfile "{{ justfile() }}" --list --unsorted

# Mock + synthetic demo in the background (no hardware, no models); stop with `just down`
up:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p {{ runtime }}
    pidfile={{ runtime }}/sherlock.pid
    log={{ runtime }}/sherlock.log
    if [[ -f $pidfile ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "already running (pid $(cat "$pidfile")); run 'just down' first" >&2
        exit 1
    fi
    rm -f "$pidfile"
    if lsof -nP -iTCP:{{ port }} -sTCP:LISTEN >/dev/null 2>&1; then
        echo "port {{ port }} is in use; set DASHBOARD_PORT to another port" >&2
        exit 1
    fi
    CAMERAS="{{ cameras }}" nohup uv run python run.py --mock --synthetic \
        --port {{ port }} >"$log" 2>&1 </dev/null &
    pid=$!
    echo "$pid" >"$pidfile"
    for _ in $(seq 1 240); do
        if curl -fsS -o /dev/null "http://localhost:{{ port }}/" 2>/dev/null; then
            echo "Sherlock demo up: http://localhost:{{ port }}/ (pid $pid, log $log)"
            echo "Stop it with: just down"
            exit 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Sherlock exited during startup; last log lines:" >&2
            tail -n 20 "$log" >&2
            rm -f "$pidfile"
            exit 1
        fi
        sleep 0.5
    done
    echo "Sherlock did not answer on port {{ port }} within 120 s; stopping it" >&2
    "{{ just_executable() }}" --justfile "{{ justfile() }}" down
    exit 1

# Stop the demo that `just up` started
down:
    #!/usr/bin/env bash
    set -euo pipefail
    pidfile={{ runtime }}/sherlock.pid
    if [[ ! -f $pidfile ]]; then
        echo "nothing to stop ({{ runtime }}/sherlock.pid not found)"
        exit 0
    fi
    pid=$(cat "$pidfile")
    # `uv run` may sit between us and python, so stop the whole process tree.
    tree() {
        local child
        for child in $(pgrep -P "$1" || true); do tree "$child"; done
        echo "$1"
    }
    # Guard against a recycled PID: only touch a process that is running run.py.
    if kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o command= | grep -q 'run\.py'; then
        pids=$(tree "$pid")
        kill $pids 2>/dev/null || true
        alive=""
        for _ in $(seq 1 40); do
            alive=""
            for p in $pids; do
                if kill -0 "$p" 2>/dev/null; then alive=1; fi
            done
            if [[ -z $alive ]]; then break; fi
            sleep 0.25
        done
        if [[ -n $alive ]]; then kill -9 $pids 2>/dev/null || true; fi
        echo "Sherlock stopped (pid $pid)"
    else
        echo "Sherlock was not running (stale pid $pid)"
    fi
    rm -f "$pidfile"

# Real cameras and models per .env, in the foreground (the yolo extra pulls in torch)
live:
    uv run --extra yolo python run.py

# Plan B on Apple silicon: Director on a dedicated Ollama, VLM on mlx-vlm, both on loopback
local-up:
    scripts/local.sh up

# Sherlock in the foreground against the local models; --synthetic rehearses with no camera
local-live *args:
    scripts/local.sh live {{args}}

# The local model servers started here, and whether they answer
local-status:
    scripts/local.sh status

# Stop exactly what `just local-up` started; fails unless both model ports are free
local-down:
    scripts/local.sh down

# Run the test suite (includes the gym optional extra on Python >= 3.13.14)
test:
    uv run --extra gym pytest

# Ruff lint gate (ruff.toml pins ruff 0.17.1 and the rule set); must report zero findings
lint:
    uv run ruff check .

# Lint, then test: what CI would run
check: lint test

# Judge-run protocol benchmark: no GPU; stub evidence stays outside results/
bench-stub *args:
    uv run scripts/bench_vllm.py --stub --metrics {{args}}

# Judge-run readiness, smoke, sweep and simulated prefix-cache A/B; no Docker
preflight-stub *args:
    uv run scripts/brev_preflight.py --stub --mode ab --requests 16 {{args}}

# Expanso Edge data plane (scripts/expanso.sh): validate the five jobs offline
expanso-check:
    scripts/expanso.sh check

# Both Expanso nodes on this machine, no Cloud; then SHERLOCK_EDGE=1 just up
expanso-local:
    scripts/expanso.sh local

# Laptop node in Expanso Cloud, and deploy all five jobs from Cloud
expanso-edge:
    scripts/expanso.sh edge

# Core node in Expanso Cloud; run this one on the Brev box
expanso-core:
    scripts/expanso.sh core

# Expanso nodes running here, and the jobs in Cloud
expanso-status:
    scripts/expanso.sh status

# Stop the Expanso nodes started on this machine
expanso-down:
    scripts/expanso.sh down

# Delete the five jobs from Expanso Cloud
expanso-clean:
    scripts/expanso.sh clean
