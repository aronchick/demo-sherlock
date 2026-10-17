#!/usr/bin/env bash
# Run on the Brev GPU box (1x H100/A100 80GB). Serves Configuration A (~32.25B generative params):
#   - Director (:8001): Nemotron 3.5 Lightning 30B-A3B (30B total / 3B active)
#   - Analyst + Verifier (:8002): shared SmolVLM 2.2B (2.25B, Idefics3 arch in vLLM)
# No speculative-decoding draft models are loaded (draft weights count toward the parameter budget).
set -euo pipefail
IMG=vllm/vllm-openai:v0.28.0
# Hopper/Ampere: use the NVFP4 repo's W4A16 kernels (~20GB). Blackwell: NVFP4 natively.
DIRECTOR=${DIRECTOR:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}
VLM=${VLM:-HuggingFaceTB/SmolVLM-Instruct}
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
REASONING_PARSER=${REASONING_PARSER:-nemotron_v3}

docker run -d --name director --gpus all --ipc=host -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface "$IMG" \
  --model "$DIRECTOR" --served-model-name "$DIRECTOR" director \
  --max-model-len 32768 --gpu-memory-utilization 0.60 --max-num-seqs 16 \
  --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser "$REASONING_PARSER"

docker run -d --name vlm --gpus all --ipc=host -p 8002:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface "$IMG" \
  --model "$VLM" --served-model-name "$VLM" vlm \
  --max-model-len 8192 --gpu-memory-utilization 0.30 --max-num-seqs 8 --limit-mm-per-prompt image=1

echo "director: http://<brev-host>:8001/v1   vlm: http://<brev-host>:8002/v1"
echo "On the Mac: set DIRECTOR_*/ANALYST_*/VERIFIER_* in .env (see .env.example, Config A block)"
