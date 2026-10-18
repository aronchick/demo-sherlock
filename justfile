# Sherlock. Needs `uv` and `just`; `uv run` syncs the environment on first use.
set shell := ["bash", "-euo", "pipefail", "-c"]

_default:
    @"{{ just_executable() }}" --justfile "{{ justfile() }}" --list --unsorted

# Ensure the configured existing GPU, model servers, tunnel and actual camera demo.
up:
    uv run --no-project scripts/brev_machine.py demo-up

# Synthetic rehearsal only; no GPU is started.
demo-up:
    uv run --no-project scripts/brev_machine.py sandbox-up

# Stop owned presenter/tunnel and verify GPU shutdown; storage may still cost.
down:
    uv run --no-project scripts/brev_machine.py demo-down

# Stop only the owned presenter; tunnel, Edge and billable GPU stay running.
down-local:
    uv run --no-project scripts/brev_machine.py local-down

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

# Both Expanso nodes on this machine, no Cloud; then SHERLOCK_EDGE=1 just demo-up
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

# Configure the existing H100 from this Mac; exact running models are reused.
brev-up instance="sherlock-nebius-h100":
    uv run --no-project scripts/brev_machine.py up {{quote(instance)}}

# Keep this terminal open: tunnel both endpoints and verify inference.
brev-connect instance="sherlock-nebius-h100":
    uv run --no-project scripts/brev_machine.py connect {{quote(instance)}}

# Run the local cameras/UI against the verified Brev tunnel, preserving .env.
brev-live *args:
    uv run --no-project scripts/brev_machine.py live -- {{args}}

# Read the actual Brev instance inventory.
brev-status:
    uv run --no-project scripts/brev_machine.py status

# Stop an explicitly named old machine only after replacement inference passes.
brev-retire old replacement:
    uv run --no-project scripts/brev_machine.py retire \
        {{quote(old)}} {{quote(replacement)}}
