#!/usr/bin/env bash
# Sherlock with every model on this Mac: Plan B when the Brev box or the venue network is out.
#   Director            nemotron-3.5-lightning:30b-mlx, a dedicated Ollama on 127.0.0.1:11436
#   Analyst + Verifier  HuggingFaceTB/SmolVLM-500M-Instruct, mlx-vlm 0.7.8 on 127.0.0.1:8002
#
#   scripts/local.sh up            start both model servers; return once each has answered
#   scripts/local.sh live [args]   Sherlock in the foreground against them (run.py args pass
#                                  through: --synthetic rehearses on two synthetic cameras,
#                                  hero and scout1); fails if the dashboard port is in use
#   scripts/local.sh env           the exports `live` uses, for: eval "$(scripts/local.sh env)"
#   scripts/local.sh status        what `up` started here, and whether it answers
#   scripts/local.sh down          stop exactly what `up` started; fail unless both ports are free
#
# Apple silicon only: mlx-vlm runs on MLX. It runs through uvx at a pinned version, so it is
# never a project dependency and Linux and Brev installs are untouched. Ollama comes from
# Homebrew (0.40 or newer, for the MLX runner); pull the Director once on a good network:
#   ollama pull nemotron-3.5-lightning:30b-mlx       (19 GB)
# Overrides: LOCAL_DIRECTOR_PORT, LOCAL_VLM_PORT, LOCAL_DIRECTOR_MODEL, LOCAL_VLM_MODEL,
# LOCAL_RUNTIME (pid and log directory), VERIFIER_MARGIN (default -1.0 here).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUNTIME="${LOCAL_RUNTIME:-$ROOT/.runtime/local}"
DIRECTOR_PORT="${LOCAL_DIRECTOR_PORT:-11436}"
VLM_PORT="${LOCAL_VLM_PORT:-8002}"
DIRECTOR_MODEL="${LOCAL_DIRECTOR_MODEL:-nemotron-3.5-lightning:30b-mlx}"
VLM_MODEL="${LOCAL_VLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}"
MLX_VLM="mlx-vlm==0.7.8"

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "$1 is not on PATH ($2)"; }
port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

# ------------------------------------------------------------ processes

# running <name> <pattern>: the recorded pid is alive and still the program we started.
running() {
  local pid
  [[ -f "$RUNTIME/$1.pid" ]] || return 1
  pid="$(cat "$RUNTIME/$1.pid")"
  kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o command= | grep -q -- "$2"
}

tree() {
  local child
  for child in $(pgrep -P "$1" || true); do tree "$child"; done
  echo "$1"
}

# stop <name> <pattern>: the whole tree, since uvx and ollama serve each run children.
stop() {
  local pid pids alive p
  [[ -f "$RUNTIME/$1.pid" ]] || return 0
  pid="$(cat "$RUNTIME/$1.pid")"
  if running "$1" "$2"; then
    pids="$(tree "$pid")"
    # shellcheck disable=SC2086 # one pid per word
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 40); do
      alive=""
      for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
      [[ -z $alive ]] && break
      sleep 0.25
    done
    # shellcheck disable=SC2086
    [[ -n $alive ]] && kill -9 $pids 2>/dev/null || true
    say "$1 stopped (pid $pid)"
  fi
  rm -f "$RUNTIME/$1.pid"
}

# wait_for <name> <pattern> <seconds> <command...>: until the command succeeds.
wait_for() {
  local name="$1" pattern="$2" secs="$3"; shift 3
  for _ in $(seq 1 "$secs"); do
    "$@" >/dev/null 2>&1 && return 0
    running "$name" "$pattern" || { tail -n 20 "$RUNTIME/$name.log" >&2; die "$name exited during startup"; }
    sleep 1
  done
  tail -n 20 "$RUNTIME/$name.log" >&2
  die "$name did not answer within $secs s"
}

director_answers() {
  curl -fsS -m 300 "http://127.0.0.1:$DIRECTOR_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"$DIRECTOR_MODEL\", \"max_tokens\": 1, \"reasoning_effort\": \"none\",
         \"messages\": [{\"role\": \"user\", \"content\": \"ok\"}]}"
}

# --------------------------------------------------------------- commands

cmd_up() {
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] ||
    die "the local models need Apple silicon (mlx-vlm runs on MLX); elsewhere use vLLM on Brev"
  need ollama "brew install ollama"
  need uvx "https://docs.astral.sh/uv/"
  local v
  v="$(ollama --version 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+' | head -n 1)"
  if [[ -z "$v" ]] || ! awk -v v="$v" 'BEGIN { split(v, a, "."); exit !(a[1] > 0 || a[2] >= 40) }'; then
    die "Ollama ${v:-?} is too old for the MLX runner; brew upgrade ollama (0.40 or newer)"
  fi
  running director "ollama serve" && die "already up; run: scripts/local.sh down"
  running vlm mlx_vlm && die "already up; run: scripts/local.sh down"
  port_busy "$DIRECTOR_PORT" && die "port $DIRECTOR_PORT is in use (set LOCAL_DIRECTOR_PORT)"
  port_busy "$VLM_PORT" && die "port $VLM_PORT is in use (set LOCAL_VLM_PORT)"
  mkdir -p "$RUNTIME"
  # A failed start stops whatever it had started, so a retry finds the ports free.
  trap '( cmd_down ) >/dev/null 2>&1 || true' EXIT

  # Dedicated instance: the shared service's default context timed out at Ollama's 5-minute
  # load limit under memory pressure; 8192 loads in seconds and an incident needs under 2k.
  OLLAMA_HOST="127.0.0.1:$DIRECTOR_PORT" OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_KEEP_ALIVE=60m \
    nohup ollama serve >|"$RUNTIME/director.log" 2>&1 </dev/null &
  echo $! >|"$RUNTIME/director.pid"
  wait_for director "ollama serve" 30 curl -fsS "http://127.0.0.1:$DIRECTOR_PORT/api/version"
  OLLAMA_HOST="127.0.0.1:$DIRECTOR_PORT" ollama list | awk 'NR > 1 { print $1 }' | grep -qx "$DIRECTOR_MODEL" ||
    die "$DIRECTOR_MODEL is not pulled; on a good network run: ollama pull $DIRECTOR_MODEL"
  # Load the Director before the VLM starts, so the two loads never compete for memory.
  say "loading $DIRECTOR_MODEL..."
  wait_for director "ollama serve" 2 director_answers

  # --top-logprobs-k 10: without it mlx-vlm sends placeholder logprobs and the Verifier falls
  # back to reading the reply's first word; with it the board shows the margin, as on Brev.
  nohup uvx --python 3.12 --from "$MLX_VLM" mlx_vlm.server --model "$VLM_MODEL" \
    --host 127.0.0.1 --port "$VLM_PORT" --top-logprobs-k 10 >|"$RUNTIME/vlm.log" 2>&1 </dev/null &
  echo $! >|"$RUNTIME/vlm.pid"
  say "loading $VLM_MODEL (first run downloads about 1 GB)..."
  wait_for vlm mlx_vlm 600 curl -fsS "http://127.0.0.1:$VLM_PORT/v1/models"
  trap - EXIT

  say "Director  http://127.0.0.1:$DIRECTOR_PORT/v1  $DIRECTOR_MODEL"
  say "VLM       http://127.0.0.1:$VLM_PORT/v1  $VLM_MODEL"
  say "Next: just local-live   (just local-live --synthetic rehearses with no camera)"
  say "Stop: just local-down"
}

cmd_env() {
  say "export DIRECTOR_BASE_URL=http://127.0.0.1:$DIRECTOR_PORT/v1"
  say "export DIRECTOR_MODEL=$DIRECTOR_MODEL"
  say "export ANALYST_BASE_URL=http://127.0.0.1:$VLM_PORT/v1"
  say "export VERIFIER_BASE_URL=http://127.0.0.1:$VLM_PORT/v1"
  say "export ANALYST_MODEL=$VLM_MODEL"
  say "export VERIFIER_MODEL=$VLM_MODEL"
  # The 500M's measured operating point on tight crops: 62/70 weapons, 0/20 cups and bottles.
  say "export VERIFIER_MARGIN=${VERIFIER_MARGIN:--1.0}"
}

# The port run.py binds: --port N or --port=N, else DASHBOARD_PORT, else 8000.
dashboard_port() {
  local port="${DASHBOARD_PORT:-8000}"
  while (($#)); do
    case "$1" in
      --port) port="${2:-}"; shift ;;
      --port=*) port="${1#--port=}" ;;
    esac
    shift
  done
  printf '%s\n' "$port"
}

cmd_live() {
  local port arg
  port="$(dashboard_port "$@")"
  port_busy "$port" && die "dashboard port $port is in use (set DASHBOARD_PORT or pass --port)"
  if ! curl -fsS -o /dev/null "http://127.0.0.1:$DIRECTOR_PORT/api/version" 2>/dev/null ||
    ! curl -fsS -o /dev/null "http://127.0.0.1:$VLM_PORT/v1/models" 2>/dev/null; then
    die "the local models are not answering; run: just local-up"
  fi
  # Exported, so they win over a .env that points at Brev (python-dotenv never overrides).
  eval "$(cmd_env)"
  for arg in "$@"; do
    # Both synthetic cameras, as `just up` runs them: the board shows two, and a covered or
    # failed hero hands the investigation to scout1. A CAMERAS already in the shell wins.
    if [[ $arg == --synthetic ]]; then
      export CAMERAS="${CAMERAS:-hero:fake:synthetic,scout1:fake:synthetic}"
    fi
  done
  exec uv run --extra yolo python run.py "$@"
}

answers() { curl -fsS -m 5 -o /dev/null "$1" 2>/dev/null && echo answering || echo "NOT answering"; }

cmd_status() {
  if running director "ollama serve"; then
    say "director: pid $(cat "$RUNTIME/director.pid"), port $DIRECTOR_PORT, $(answers "http://127.0.0.1:$DIRECTOR_PORT/api/version")"
  else say "director: not running here"; fi
  if running vlm mlx_vlm; then
    say "vlm: pid $(cat "$RUNTIME/vlm.pid"), port $VLM_PORT, $(answers "http://127.0.0.1:$VLM_PORT/v1/models")"
  else say "vlm: not running here"; fi
}

cmd_down() {
  local p ports busy=""
  stop vlm mlx_vlm
  stop director "ollama serve"
  for p in "$VLM_PORT" "$DIRECTOR_PORT"; do
    for _ in $(seq 1 20); do port_busy "$p" || continue 2; sleep 0.25; done
    busy+=" $p"
  done
  ports="${busy# }"
  [[ -z "$busy" ]] || die "still in use:$busy by $(lsof -nP -iTCP:"${ports// /,}" -sTCP:LISTEN 2>/dev/null |
    awk 'NR > 1 { print $1 "/" $2 }' | sort -u | tr '\n' ' ')"
  say "ports $DIRECTOR_PORT and $VLM_PORT are free"
}

case "${1:-}" in
  up) cmd_up ;;
  live) shift; cmd_live "$@" ;;
  env) cmd_env ;;
  status) cmd_status ;;
  down) cmd_down ;;
  *) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
