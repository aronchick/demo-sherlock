# Sherlock — a camera that investigates

Small models that notice, decide, zoom, read, and verify. NVIDIA × Red Hat Small Models Hack, Raleigh, Oct 2026.
Tracks: Multi-Agent (primary), Tool-Calling, Nemotron + vLLM bounties.

```
Scout (YOLOv8n detector, every frame, ~10 ms)
  └─ new object? ──► Director (Nemotron 3.5 Lightning 30B-A3B + camera tools)
                        ├─ look_at / move  → PTZ camera (OBSBOT or software crop)
                        ├─ snapshot        → Analyst (SmolVLM 2.2B on the zoomed crop only)
                        └─ report          → Verifier (same SmolVLM 2.2B instance: does the crop support the claim?)
```
One tiny model gates the expensive ones. The dashboard counts "VLM calls avoided" live.

## Model lineup

### Configuration A (default, ~32.25B generative parameters)
Every generative model runs on vLLM (`vllm/vllm-openai:v0.28.0`), covering both the NVIDIA Nemotron Lightning bounty and the Red Hat vLLM bounty. Speculative-decoding draft models (DSpark, DFlash, MTP) are disabled because draft weights count toward the parameter budget.

| Role | Hugging Face Repo ID | Verified Parameters | Serving Details |
| :--- | :--- | :--- | :--- |
| **Director** | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | **30B total / 3B active** on model card (the BF16 safetensors metadata reports `31,577,937,344` including 270 embedded `mtp.layers.*` speculative draft tensors that vLLM does not load when speculative decoding is off) | vLLM on `:8001`, `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3`, thinking OFF by default (`DIRECTOR_THINKING=0` passes `chat_template_kwargs: {"enable_thinking": false}`) |
| **Analyst + Verifier** | `HuggingFaceTB/SmolVLM-Instruct` | **2.25B** (`2,246,272,880` safetensors parameters; `Idefics3ForConditionalGeneration` in vLLM v0.28.0) | Single shared vLLM instance on `:8002` (counted once). `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (`2,246,784,880` params, `SmolVLMForConditionalGeneration`) is also supported |
| **Scout** | `yolov8n.pt` (Ultralytics YOLOv8n) | **3.2M** (excluded) | Non-generative detector; excluded from the generative parameter total |
| **Total (generative)** | | **32.25B** (30B + 2.25B, under the 33B hard cap) | Zero draft models |

*Why SmolVLM 2.2B instead of Moondream 2 (1.9B):* `vikhyatk/moondream2` (`1,927,237,104` parameters, `HfMoondream` / `moondream1`) is not in vLLM v0.28.0's supported models list (`vllm` v0.28.0 only implements `Moondream3ForCausalLM` for the 9B MoE `moondream/moondream3-preview`). `HuggingFaceTB/SmolVLM-Instruct` runs natively in vLLM v0.28.0 via `Idefics3ForConditionalGeneration`.

### Configuration B (NeMo Gym ablation, ~17.6B–21.0B generative parameters)
Configuration B swaps the Director and shared VLM via `.env` (see the commented-out Configuration B block in `.env.example`) for the NeMo Gym ablation:
- **Director**: `nvidia/NVIDIA-Nemotron-Nano-9B-v2` (**8.89B**, `8,888,227,328` safetensors parameters) or `nvidia/NVIDIA-Nemotron-Nano-12B-v2` (**12.31B**, `12,310,001,152` safetensors parameters).
- **Analyst + Verifier (shared)**: `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` (**8.72B**, `8,721,768,454` safetensors parameters; Llama-3.1-8B-Instruct + C-RADIOv2-H vision encoder, supported in vLLM v0.28.0 via `Llama_Nemotron_Nano_VL`).
- **Total generative parameters**: **17.61B** (with Nano 9B v2) or **21.03B** (with Nano 12B v2).
- **Tradeoff**: Trades the Nemotron 3.5 Lightning bounty for a larger 8.72B VLM on zoomed crops while staying well under budget. Do not combine Nemotron 3.5 Lightning (30B) with a 7–8B VLM (~38.7B total, over budget).

## Run with nothing (sandbox, airplane, dead wifi)
```bash
pip install -r requirements.txt            # ultralytics optional for this mode
CAMERAS=hero:fake:synthetic,scout1:fake:synthetic python run.py --mock --synthetic
open http://localhost:8000
```
Rule-based director, mock VLMs, generated video. Press **Break the next tool call** and **Cover hero lens** on the dashboard to see recovery.

## vLLM on Brev (Config A default)
1. On the Brev GPU box (1x H100/A100 80GB), start both vLLM containers:
```bash
./brev/serve.sh
```
2. On the Mac, copy `.env.example` to `.env` and point at the Brev host (or local SSH tunnel on `:8001` and `:8002`):
```bash
cp .env.example .env              # CAMERAS=hero:fake:0 uses the laptop webcam with software PTZ
python run.py
```
If the endpoint is unreachable mid-pitch, the Director automatically drops to the rule-based fallback (`RuleDirector`).

## When the OBSBOT arrives
1. `python scripts/ptz_probe.py` — checks whether the camera answers standard UVC pan/tilt/zoom.
2. If yes: `CAMERAS=hero:obsbot:0,scout1:fake:1`. Nothing else changes.
3. If no: Plan B is OSC via OBSBOT Center; implement `ObsbotCamera.move` against it.

## Layout
```
sherlock/camera/     Camera interface, FakePTZ (crop/zoom), ObsbotCamera (UVC), SyntheticSource
sherlock/agents/     scout.py (YOLO + motion fallback + dedupe/dwell), director.py (tool loop + RuleDirector), vision.py (Analyst, Verifier, mocks)
sherlock/tools.py    Tool schemas + dispatcher + failure injection
sherlock/orchestrator.py   Scout → investigate → verify loop, stats
sherlock/dashboard/  FastAPI SSE + MJPEG + the judge-facing UI
tests/               End-to-end mock run incl. tool failure recovery and covered-lens fallback
gym/                 Native NeMo Gym resources server, verifier, agent, and YAML ablation configs
results/             Per-ablation JSON artifacts (results/ablation_*.json)
```

## Development
The lint gate is one command, and it must report zero findings:
```bash
uvx ruff@0.17.1 check .
```
`ruff.toml` pins this exact version (`required-version`; other versions refuse to run) and spells out its rule set, so every machine gets the same answer.

## NeMo Gym ablation

Sherlock includes a native [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) (`nemo-gym==0.7.0`) environment and multi-metric verifier under `gym/` for single-component agent ablations. The 8 scripted tasks (`gym/resources_servers/sherlock/data/tasks.jsonl`) and the verifier (`SherlockResourcesServer.verify`) stay fixed while YAML configs swap one Director agent component at a time:

1. **System prompt (`prompt_v1` vs `nemotron_lightning`)**: `gym/prompts/system_v1.yaml` vs `gym/prompts/system_v2.yaml` (adds zero-snapshot benign filtering and the two-crop weapon + head/face protocol).
2. **Tool descriptions (`terse_tools` vs `nemotron_lightning`)**: `terse` (3-word tool strings, stripped parameter hints) vs `verbose` (`sherlock.tools.TOOL_SCHEMAS`).
3. **Reasoning mode (`thinking_on` vs `nemotron_lightning`)**: `enable_thinking: true` vs `enable_thinking: false` on `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`.
4. **Director model (`qwen_director` vs `nemotron_lightning`)**: `Qwen/Qwen2.5-7B-Instruct` (7.6B) vs `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (30B total / 3B active).

Every configuration pairs the Director with `HuggingFaceTB/SmolVLM-Instruct` (2.25B) for the Analyst/Verifier VLM, keeping total generative parameters at or under the 33B limit (32.25B for Nemotron 3.5 Lightning configs; 9.85B for Qwen 2.5 7B).

### Run one ablation with one command

```bash
# Local end-to-end validation (uses the built-in OpenAI-compatible stub when DIRECTOR_BASE_URL is unset):
uv run python -m gym.run_ablation --config gym/configs/ablation_nemotron_lightning.yaml

# Run against a live Brev vLLM or Ollama endpoint:
DIRECTOR_BASE_URL=https://<brev-host>/v1 uv run python -m gym.run_ablation --config gym/configs/ablation_nemotron_lightning.yaml

# Run all 5 YAML ablations and write results/ablation_*.json:
uv run python -m gym.run_ablation --all
```

### Ablation metrics (local stub validation; replace with live Brev vLLM numbers once `DIRECTOR_BASE_URL` is up)

| Ablation Config | Swapped Component | Director Model | Total Gen Params | Prompt | Tools | Thinking | Tool-Call Validity | Task Success | Steps to Report | Time to Report (ms) |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|
| `prompt_v1` | System prompt (`v1`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 32.25B | `system_v1` | `verbose` | `off` | 100.0% | 50.0% | 3.50 | 49.5 *(stub)* |
| `terse_tools` | Tool schemas (`terse`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 32.25B | `system_v2` | `terse` | `off` | 88.7% | 100.0% | 4.12 | 57.0 *(stub)* |
| `thinking_on` | Thinking (`on`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 32.25B | `system_v2` | `verbose` | `on` | 100.0% | 87.5% | 3.75 | 151.1 *(stub)* |
| `qwen_director` | Model (`Qwen2.5-7B`) | `Qwen2.5-7B-Instruct` | 9.85B | `system_v2` | `verbose` | `off` | 100.0% | 87.5% | 3.38 | 76.3 *(stub)* |
| `nemotron_lightning` | Reference (`v2` + `verbose` + `off`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 32.25B | `system_v2` | `verbose` | `off` | 100.0% | 100.0% | 3.50 | 60.9 *(stub)* |

## Demo script (3 min)
1. Dashboard up, wide view showing the room, counter climbing with zero VLM calls.
2. Judge holds up an object. Scout flags it, camera whirs over and zooms, Analyst reads the label, Verifier confirms. Point at latency per stage.
3. Press **Break the next tool call** → Director gets an error, retries differently, still reports.
4. Cover the hero lens → Director lists cameras, finishes on scout1.
5. Close on the ratio: frames handled by a 10 ms model vs. VLM calls actually made.
