#!/usr/bin/env bash
# Sherlock's Expanso Edge data plane. Two nodes, five jobs (jobs/*.yaml):
#   sherlock-edge (the laptop): E1 telemetry, E2 incidents, E3 examples
#   sherlock-core (the Brev box): C1 core intake, C2 vLLM metrics
#
#   scripts/expanso.sh check     validate the five jobs offline
#   scripts/expanso.sh local     both nodes on this machine, no Cloud (rehearsal)
#   scripts/expanso.sh edge      laptop node in Expanso Cloud; deploys all five jobs
#   scripts/expanso.sh core      core node in Expanso Cloud (run this on the Brev box)
#   scripts/expanso.sh status    nodes started here, and Cloud jobs when configured
#   scripts/expanso.sh down      stop the nodes started on this machine
#   scripts/expanso.sh clean     delete the five jobs from Expanso Cloud
#
# Credentials are read from one owner-only env file, never a global profile:
# EXPANSO_WORKSPACE_ENV, else the demos' shared _demo-kit/.env, else ./.env.
# `edge` needs EXPANSO_CLI_ENDPOINT, EXPANSO_CLI_API_KEY and
# EXPANSO_EDGE_BOOTSTRAP_TOKEN; `core` needs only the bootstrap token.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUNTIME="$ROOT/.runtime/expanso"
STATE="$ROOT/.cloud-state"
EDGE_JOBS=(jobs/sherlock-telemetry.yaml jobs/sherlock-incidents.yaml jobs/sherlock-examples.yaml)
CORE_JOBS=(jobs/sherlock-core-intake.yaml jobs/sherlock-vllm-metrics.yaml)
JOB_NAMES=(sherlock-telemetry sherlock-incidents sherlock-examples sherlock-core-intake sherlock-vllm-metrics)
EDGE_API=127.0.0.1:23110
CORE_API=127.0.0.1:23120
EDGE_PORTS=(23111 23112 23113)
CORE_PORTS=(23121)

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "$1 is not on PATH (${2:-see README: Expanso})"; }

# ------------------------------------------------------------ credentials

env_file() {
  if [[ -n "${EXPANSO_WORKSPACE_ENV:-}" ]]; then printf '%s' "$EXPANSO_WORKSPACE_ENV"; return; fi
  local common kit
  common="$(git rev-parse --git-common-dir 2>/dev/null || true)"
  if [[ -n "$common" ]]; then
    kit="$(cd "$common/../.." 2>/dev/null && pwd)/_demo-kit/.env"
    [[ -f "$kit" ]] && { printf '%s' "$kit"; return; }
  fi
  printf '%s' "$ROOT/.env"
}

# Reads only the three Expanso keys; never sources the file.
load_creds() {
  local file perm key value
  file="$(env_file)"
  [[ -f "$file" ]] || die "no credentials file at $file (README: Expanso)"
  perm="$(stat -f '%Lp' "$file" 2>/dev/null || stat -c '%a' "$file")"
  [[ "$perm" == 600 ]] || die "$file must be readable only by you (chmod 600)"
  EXPANSO_CLI_ENDPOINT="" EXPANSO_CLI_API_KEY="" EXPANSO_EDGE_BOOTSTRAP_TOKEN=""
  while IFS='=' read -r key value; do
    key="${key// /}"; value="${value%$'\r'}"; value="${value#\"}"; value="${value%\"}"
    case "$key" in
      EXPANSO_CLI_ENDPOINT) EXPANSO_CLI_ENDPOINT="$value" ;;
      EXPANSO_CLI_API_KEY) EXPANSO_CLI_API_KEY="$value" ;;
      EXPANSO_EDGE_BOOTSTRAP_TOKEN) EXPANSO_EDGE_BOOTSTRAP_TOKEN="$value" ;;
    esac
  done < <(grep -E '^ *EXPANSO_[A-Z_]+ *=' "$file" || true)
}

require_cli_creds() {
  [[ "$EXPANSO_CLI_ENDPOINT" =~ ^https://[^/]+$ ]] \
    || die "EXPANSO_CLI_ENDPOINT should look like https://<workspace>.us1.cloud.expanso.io:9010"
  [[ "$EXPANSO_CLI_API_KEY" == exp_ak_* ]] || die "EXPANSO_CLI_API_KEY should start with exp_ak_"
}

require_bootstrap_token() {
  [[ "$EXPANSO_EDGE_BOOTSTRAP_TOKEN" == exp_bk_* ]] \
    || die "EXPANSO_EDGE_BOOTSTRAP_TOKEN should start with exp_bk_"
}

# The key travels in the environment, never on the command line, and the CLI
# keeps its config inside this repository's ignored runtime folder.
cloud_cli() {
  (
    unset EXPANSO_CLI_PROFILE
    EXPANSO_CLI_CONFIG_DIR="$RUNTIME/cli" EXPANSO_CLI_AUTH_API_KEY="$EXPANSO_CLI_API_KEY" \
      expanso-cli --endpoint "$EXPANSO_CLI_ENDPOINT" "$@"
  )
}

local_cli() { expanso-cli --endpoint "http://$1" "${@:2}"; }

# ------------------------------------------------------------------ nodes

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

node_running() {
  local pid
  [[ -f "$RUNTIME/$1.pid" ]] || return 1
  pid="$(cat "$RUNTIME/$1.pid")"
  kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o command= | grep -q expanso-edge
}

# The pipelines read these at start; ${VAR:default} in jobs/*.yaml.
node_env() {
  if [[ "$1" == edge ]]; then
    export EDGE_NODE_NAME=sherlock-edge
    export SHERLOCK_SPOOL_DIR="${SHERLOCK_SPOOL_DIR:-$RUNTIME/spool}"
    mkdir -p "$SHERLOCK_SPOOL_DIR"
  else
    export EDGE_NODE_NAME=sherlock-core
    export SHERLOCK_CORE_DIR="${SHERLOCK_CORE_DIR:-$RUNTIME/core}"
    mkdir -p "$SHERLOCK_CORE_DIR"
  fi
}

# start_node <edge|core> <local|cloud>
start_node() {
  local role="$1" mode="$2" api ports args
  if [[ "$role" == edge ]]; then api="$EDGE_API"; ports=("${EDGE_PORTS[@]}"); else api="$CORE_API"; ports=("${CORE_PORTS[@]}"); fi
  node_running "$role" && die "sherlock-$role is already running here; run: scripts/expanso.sh down"
  for p in "${api##*:}" "${ports[@]}"; do port_busy "$p" && die "port $p is in use"; done
  mkdir -p "$RUNTIME"
  node_env "$role"
  args=(run -c "edge/sherlock-$role.yaml" --no-watch --api-listen "$api")
  if [[ "$mode" == local ]]; then
    rm -rf "$RUNTIME/$role-local"
    args+=(--local --data-dir "$RUNTIME/$role-local")
  else
    args+=(--data-dir "$STATE/sherlock-$role")
  fi
  nohup expanso-edge "${args[@]}" >| "$RUNTIME/$role.log" 2>&1 </dev/null &
  echo $! >| "$RUNTIME/$role.pid"
  echo "$mode" >| "$RUNTIME/mode"
}

wait_api() {
  for _ in $(seq 1 60); do
    local_cli "$2" node list >/dev/null 2>&1 && return 0
    node_running "$1" || { tail -n 20 "$RUNTIME/$1.log" >&2; die "sherlock-$1 exited"; }
    sleep 1
  done
  die "sherlock-$1 API did not answer on $2"
}

wait_ports() {
  local role="$1"; shift
  for p in "$@"; do
    for _ in $(seq 1 60); do port_busy "$p" && continue 2; sleep 1; done
    tail -n 20 "$RUNTIME/$role.log" >&2
    die "a sherlock-$role pipeline is not listening on port $p"
  done
}

# deploy <cli function and args...> -- <job files...>; an unchanged job is restarted.
deploy() {
  local cli=() file name out
  while [[ "$1" != -- ]]; do cli+=("$1"); shift; done; shift
  for file in "$@"; do
    name="$(awk '/^name:/ {print $2; exit}' "$file")"
    if out="$("${cli[@]}" job deploy "$file" 2>&1)"; then
      say "  deployed $name"
    elif grep -q NO_CHANGES_DETECTED <<<"$out"; then
      "${cli[@]}" job rerun "$name" >/dev/null && say "  $name unchanged; restarted"
    else
      printf '%s\n' "$out" | sed -n '1,8p' >&2
      die "could not deploy $file"
    fi
  done
}

bootstrap() {
  local role="$1" dir="$STATE/sherlock-$1"
  mkdir -p "$dir" && chmod 700 "$STATE" "$dir"
  [[ -f "$dir/auth/credentials.creds" ]] && return 0
  say "registering this machine as sherlock-$role in your Expanso Cloud workspace..."
  EXPANSO_EDGE_BOOTSTRAP_TOKEN="$EXPANSO_EDGE_BOOTSTRAP_TOKEN" \
    expanso-edge bootstrap -c "edge/sherlock-$role.yaml" --data-dir "$dir" \
    >| "$RUNTIME/bootstrap-$role.log" 2>&1 || {
      tail -n 8 "$RUNTIME/bootstrap-$role.log" >&2
      die "bootstrap failed; the token may be expired or revoked"
    }
}

stop_node() {
  local role="$1" pid
  [[ -f "$RUNTIME/$role.pid" ]] || return 0
  pid="$(cat "$RUNTIME/$role.pid")"
  if node_running "$role"; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
    kill -9 "$pid" 2>/dev/null || true
    say "sherlock-$role stopped"
  fi
  rm -f "$RUNTIME/$role.pid"
}

# --------------------------------------------------------------- commands

cmd_check() {
  need expanso-edge
  expanso-edge validate "${EDGE_JOBS[@]}" "${CORE_JOBS[@]}"
}

cmd_local() {
  need expanso-edge; need expanso-cli
  cmd_check >/dev/null
  # Locally the core pushes serving numbers straight to the dashboard.
  export SHERLOCK_DASHBOARD_URL="${SHERLOCK_DASHBOARD_URL:-http://127.0.0.1:${DASHBOARD_PORT:-8000}}"
  start_node core local
  start_node edge local
  wait_api core "$CORE_API"
  wait_api edge "$EDGE_API"
  deploy local_cli "$CORE_API" -- "${CORE_JOBS[@]}"
  deploy local_cli "$EDGE_API" -- "${EDGE_JOBS[@]}"
  wait_ports core "${CORE_PORTS[@]}"
  wait_ports edge "${EDGE_PORTS[@]}"
  say "Expanso Edge up in local mode (no Cloud): both nodes on this machine."
  say "Core records: $SHERLOCK_CORE_DIR    spool: $SHERLOCK_SPOOL_DIR"
  say "Run Sherlock with the sink on:  SHERLOCK_EDGE=1 just up   (mock)  or  SHERLOCK_EDGE=1 just live"
}

cloud_node_state() {
  cloud_cli node list -f json -L "role=sherlock-$1" 2>/dev/null \
    | jq -r --arg name "sherlock-$1" '
        [.[] | select(.spec.name == $name) | .status.connection_state]
        | if index("connected") then "connected" else (first // "") end' 2>/dev/null || true
}

wait_connected() {
  local state=""
  say "waiting for sherlock-$1 to connect to Expanso Cloud..."
  for _ in $(seq 1 90); do
    state="$(cloud_node_state "$1")"
    [[ "$state" == connected ]] && { say "sherlock-$1 connected"; return 0; }
    node_running "$1" || { tail -n 20 "$RUNTIME/$1.log" >&2; die "sherlock-$1 exited"; }
    sleep 2
  done
  die "sherlock-$1 did not show as connected (last state: ${state:-none})"
}

cmd_edge() {
  need expanso-edge; need expanso-cli; need jq
  cmd_check >/dev/null
  load_creds; require_cli_creds; require_bootstrap_token
  if node_running edge; then
    wait_connected edge
    wait_ports edge "${EDGE_PORTS[@]}"
    say "sherlock-edge already connected; reusing active Cloud pipelines"
    return
  fi
  mkdir -p "$RUNTIME"
  cloud_cli node list >/dev/null 2>"$RUNTIME/cloud-check.log" || {
    sed -n '1,6p' "$RUNTIME/cloud-check.log" >&2
    die "Expanso Cloud rejected the endpoint or API key"
  }
  bootstrap edge
  start_node edge cloud
  wait_connected edge
  say "deploying all five jobs from Expanso Cloud (the core jobs start when sherlock-core connects)..."
  deploy cloud_cli -- "${EDGE_JOBS[@]}" "${CORE_JOBS[@]}"
  wait_ports edge "${EDGE_PORTS[@]}"
  say "Run Sherlock with the sink on:  SHERLOCK_EDGE=1 just live"
}

cmd_core() {
  need expanso-edge "install it: curl -fsSL https://get.expanso.io/edge/install.sh | bash"
  if node_running core; then
    say "sherlock-core already running; reusing its Cloud connection"
    return
  fi
  if [[ ! -f "$STATE/sherlock-core/auth/credentials.creds" ]]; then
    load_creds; require_bootstrap_token
  fi
  # The dashboard is reached through the laptop tunnel's reverse port.
  export SHERLOCK_DASHBOARD_URL="${SHERLOCK_DASHBOARD_URL:-http://127.0.0.1:23100}"
  export SHERLOCK_CORE_DIR="${SHERLOCK_CORE_DIR:-$ROOT/.runtime/expanso/core}"
  mkdir -p "$RUNTIME"
  bootstrap core
  start_node core cloud
  say "sherlock-core started; C1 and C2 start once Expanso Cloud assigns them."
  say "Records land in $SHERLOCK_CORE_DIR. Check with: scripts/expanso.sh status"
}

cmd_status() {
  local role
  for role in edge core; do
    if node_running "$role"; then say "sherlock-$role: running here ($(cat "$RUNTIME/mode" 2>/dev/null || echo ?))"
    else say "sherlock-$role: not running here"; fi
  done
  load_creds 2>/dev/null || return 0
  [[ "$EXPANSO_CLI_API_KEY" == exp_ak_* ]] || return 0
  cloud_cli node list -L demo=sherlock 2>/dev/null || true
  cloud_cli job list 2>/dev/null | grep -E 'NAME|sherlock-' || true
}

cmd_down() {
  stop_node edge
  stop_node core
  rm -f "$RUNTIME/mode"
}

cmd_clean() {
  need expanso-cli
  load_creds; require_cli_creds
  local name
  for name in "${JOB_NAMES[@]}"; do
    cloud_cli job delete "$name" >/dev/null 2>&1 && say "  deleted $name" || true
  done
}

case "${1:-}" in
  check) cmd_check ;;
  local) cmd_local ;;
  edge) cmd_edge ;;
  core) cmd_core ;;
  status) cmd_status ;;
  down) cmd_down ;;
  clean) cmd_clean ;;
  *) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
