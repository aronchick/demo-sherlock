#!/usr/bin/env bash
# Run on the Brev GPU box (1x H100/A100 80GB). Serves Configuration A by default, or swaps the
# Director container to Qwen/Qwen2.5-7B-Instruct when DIRECTOR_PRESET=qwen.
# Generative params per run (never co-resident):
#   - Default (DIRECTOR_PRESET=lightning):
#     30.51B by model card (30B + 0.51B); 32.09B by checkpoint tensor count (31.58B + 0.51B, under the 33B cap).
#       - Director (:8001): Nemotron 3.5 Lightning 30B-A3B (30B total / 3B active by card; BF16 shard headers:
#         31.58B without MTP, 32.91B with its 270 embedded mtp.* tensors, which vLLM skips without a
#         speculative config; see README "Parameter budget and speculation")
#       - Analyst + Verifier (:8002): shared SmolVLM-500M (507,482,304 params, Idefics3 arch in vLLM)
#   - Qwen ablation (DIRECTOR_PRESET=qwen):
#     8.12B total (7.62B Qwen/Qwen2.5-7B-Instruct + 0.51B SmolVLM-500M-Instruct, <= 33B by both card and tensors).
#     Stops/removes the Lightning `director` container first, then starts `Qwen/Qwen2.5-7B-Instruct` on :8001
#     with `--enable-auto-tool-choice --tool-call-parser hermes` (verified from vLLM v0.28.0 tool-calling docs).
# No speculative-decoding draft models are loaded (draft weights count toward the parameter budget).
set -euo pipefail
IMG=vllm/vllm-openai:v0.28.0
DIRECTOR_PRESET=${DIRECTOR_PRESET:-lightning}

if [[ "$DIRECTOR_PRESET" == "qwen" ]]; then
  DIRECTOR=${DIRECTOR:-Qwen/Qwen2.5-7B-Instruct}
  TOOL_PARSER=${TOOL_PARSER:-hermes}
  REASONING_PARSER=${REASONING_PARSER:-none}
  if docker container inspect director >/dev/null 2>&1; then
    docker container stop --timeout 30 director >/dev/null
    docker container rm director >/dev/null
  fi
else
  # Hopper/Ampere: use the NVFP4 repo's W4A16 kernels (~20GB). Blackwell: NVFP4 natively.
  DIRECTOR=${DIRECTOR:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}
  TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
  REASONING_PARSER=${REASONING_PARSER:-nemotron_v3}
fi

VLM=${VLM:-HuggingFaceTB/SmolVLM-500M-Instruct}
PREFIX_CACHING=${PREFIX_CACHING:-1}
if [[ "$PREFIX_CACHING" == "1" ]]; then
  PREFIX_CACHE_FLAGS=(--enable-prefix-caching --prefix-caching-hash-algo sha256)
else
  PREFIX_CACHE_FLAGS=(--no-enable-prefix-caching)
fi

REASONING_FLAGS=()
if [[ -n "$REASONING_PARSER" && "$REASONING_PARSER" != "none" ]]; then
  REASONING_FLAGS=(--reasoning-parser "$REASONING_PARSER")
fi

docker run -d --name director --gpus all --ipc=host -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface "$IMG" \
  --model "$DIRECTOR" --served-model-name "$DIRECTOR" director \
  --max-model-len 32768 --gpu-memory-utilization 0.60 --max-num-seqs 16 \
  "${PREFIX_CACHE_FLAGS[@]}" \
  --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" "${REASONING_FLAGS[@]}"

if [[ "$DIRECTOR_PRESET" != "qwen" ]] || ! docker container inspect vlm >/dev/null 2>&1; then
  docker run -d --name vlm --gpus all --ipc=host -p 8002:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface "$IMG" \
    --model "$VLM" --served-model-name "$VLM" vlm \
    --max-model-len 8192 --gpu-memory-utilization 0.30 --max-num-seqs 8 --limit-mm-per-prompt '{"image": 1}'
fi

echo "director ($DIRECTOR_PRESET): http://<brev-host>:8001/v1   vlm: http://<brev-host>:8002/v1"
if [[ "$DIRECTOR_PRESET" == "qwen" ]]; then
  echo "For NeMo Gym Qwen ablation: set QWEN_DIRECTOR_BASE_URL=http://<brev-host>:8001/v1"
else
  echo "On the Mac: set DIRECTOR_*/ANALYST_*/VERIFIER_* in .env (see .env.example, Config A block)"
fi
