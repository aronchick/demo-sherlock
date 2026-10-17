#!/usr/bin/env bash
# Run by just brev-up inside a source-only release on the selected H100.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$PWD/.runtime/tools:$HOME/.local/bin:$PATH"
DIRECTOR=RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8
VLM=HuggingFaceTB/SmolVLM-500M-Instruct
IMG=vllm/vllm-openai:v0.28.0
export DIRECTOR VLM

for tool in docker nvidia-smi curl; do
  command -v "$tool" >/dev/null || { echo "Missing $tool" >&2; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
docker info >/dev/null
mkdir -p .runtime
if ! command -v uv >/dev/null; then
  curl --fail --location --silent --show-error \
    https://astral.sh/uv/0.12.17/install.sh -o .runtime/install-uv.sh
  UV_INSTALL_DIR="$PWD/.runtime/tools" UV_NO_MODIFY_PATH=1 \
    sh .runtime/install-uv.sh
fi

# Validate both names before stopping either. Preserve HF downloads and other containers.
reuse=1
for name in director vlm; do
  expected="$DIRECTOR"
  [[ "$name" == vlm ]] && expected="$VLM"
  if ! docker container inspect "$name" >/dev/null 2>&1; then
    reuse=0
    continue
  fi
  image=$(docker inspect --format '{{.Config.Image}}' "$name")
  args=$(docker inspect --format '{{range .Config.Cmd}}{{println .}}{{end}}' "$name")
  model=$(printf '%s\n' "$args" | awk 'previous == "--model" {print; exit} {previous=$0}')
  case "$name:$model" in
    director:nvidia/NVIDIA-Nemotron-*|director:RedHatAI/NVIDIA-Nemotron-*|director:Qwen/Qwen2.5-7B-Instruct|vlm:HuggingFaceTB/SmolVLM-500M-Instruct) ;;
    *) echo "Refusing to replace unrecognized $name model: $model" >&2; exit 1 ;;
  esac
  [[ "$image" == vllm/vllm-openai:* ]] || {
    echo "Refusing to replace $name from image $image" >&2; exit 1;
  }
  running=$(docker inspect --format '{{.State.Running}}' "$name")
  if [[ "$model" != "$expected" || "$image" != "$IMG" || "$running" != true ]]; then
    reuse=0
  fi
done

if [[ "$reuse" == 1 ]]; then
  echo 'Exact FP8 and VLM containers already running; checking readiness without restart.'
else
  # Download before disrupting working services. Record arguments, not credentials.
  docker pull "$IMG"
  for name in director vlm; do
    if docker container inspect "$name" >/dev/null 2>&1; then
      docker inspect --format '{{json .Config.Cmd}}' "$name" > ".runtime/previous-$name-args.json"
      docker container stop --timeout 30 "$name"
      docker container rm "$name"
    fi
  done
  DIRECTOR_PRESET=lightning TOOL_PARSER=qwen3_coder \
    REASONING_PARSER=nemotron_v3 PREFIX_CACHING=1 \
    MODEL_BIND_HOST=127.0.0.1 bash brev/serve.sh
fi

export DIRECTOR_BASE_URL=http://127.0.0.1:8001/v1
export ANALYST_BASE_URL=http://127.0.0.1:8002/v1
export VERIFIER_BASE_URL="$ANALYST_BASE_URL"
export DIRECTOR_MODEL="$DIRECTOR"
export ANALYST_MODEL="$VLM" VERIFIER_MODEL="$VLM"
export DIRECTOR_API_KEY=EMPTY ANALYST_API_KEY=EMPTY
export VERIFIER_API_KEY=EMPTY OPENAI_API_KEY=EMPTY
uv run scripts/brev_preflight.py --mode smoke --ready-timeout 1800 \
  --output-dir .runtime/preflight
