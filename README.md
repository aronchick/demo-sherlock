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
Rule-based director, mock VLMs, a scripted room on a 26 s loop: someone holds up a cup (ignored, 0 VLM calls),
then someone holds up a knife (look_at → snapshot → look_at head → face snapshot → verify → alert with both crops).
Press **Break the next tool call**, **Corrupt next tool args** or **Cover hero lens** on the dashboard to see recovery.
Evidence lands in `snapshots/<incident>/` (`EVIDENCE_DIR` overrides). The face crop is stored for a human only:
no model ever sees it, and nothing identifies or describes the person.

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
sherlock/orchestrator.py   Scout → investigate → verify loop, per-incident evidence, stats
sherlock/scene.py    Scripted synthetic room (cup, then knife) for --synthetic
sherlock/dashboard/  FastAPI SSE + MJPEG + the judge-facing UI
tests/               End-to-end mock run: alert chain, ignore path (zero VLM calls), evidence, recovery
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
| `prompt_v1` | System prompt (`v1`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 32.25B | `system_v1` | `verbose` | `off` | 100.0% | 50.0% | 4.50 | 65.9 *(stub)* |
| `terse_tools` | Tool schemas (`terse`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 32.25B | `system_v2` | `terse` | `off` | 90.5% | 100.0% | 4.88 | 71.8 *(stub)* |
| `thinking_on` | Thinking (`on`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 32.25B | `system_v2` | `verbose` | `on` | 100.0% | 87.5% | 4.62 | 198.3 *(stub)* |
| `qwen_director` | Model (`Qwen2.5-7B`) | `Qwen2.5-7B-Instruct` | 9.85B | `system_v2` | `verbose` | `off` | 100.0% | 87.5% | 4.12 | 86.2 *(stub)* |
| `nemotron_lightning` | Reference (`v2` + `verbose` + `off`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 32.25B | `system_v2` | `verbose` | `off` | 100.0% | 100.0% | 4.25 | 65.9 *(stub)* |

## Demo script (3 min)

Use a red fedora with a black band and a black Red Hat / NVIDIA baseball cap.
The fedora gives the audience an easy first object to recognize; the cap adds
small lettering that gives the camera a reason to zoom.

### Hat setup and rehearsal

- Set `SCOUT_BACKEND=motion` for this sequence, with real Analyst and Verifier
  endpoints. The default `yolov8n.pt` model has no hat class. The existing motion
  scout can flag a newly introduced prop without knowing what it is; describe
  its trigger as motion until the Analyst identifies the object.
- Start with the props out of frame so the motion scout learns the background.
  Introduce one hat at a time, hold it still through the investigation, and keep
  the cap's lettering facing the camera. Between takes, remove the prop for at
  least four seconds so the Scout's three-second tracking memory can expire.
- Rehearse at the actual camera distance. Check that the crop contains the hat,
  that the Analyst can read each name it reports, and that the Verifier checks
  the resulting claim. Red fabric alone does not establish a Red Hat brand.
  If the cap lettering is unreadable, report that uncertainty. Mock mode only
  exercises the scripted cup-and-knife room and does not validate either hat.

### Presenter sequence

1. **0:00–0:20, wide view.** Show the quiet scene and the frame count before
   introducing either prop. “Sherlock watches for a change before asking the
   vision models to inspect a crop.”
2. **0:20–1:00, red fedora.** Bring it into view and hold it still. Follow the
   motion event, camera movement or software crop, Analyst description, and
   Verifier verdict. A suitable claim, if supported by the crop, is “a red hat
   with a black band.” Point at the per-stage latency.
3. **1:00–1:40, black cap.** Remove the fedora, pause, then introduce the cap
   with the lettering forward. “This hat is black. Let's see what the lettering
   says.” Show the crop and actual result; name Red Hat and NVIDIA only if the
   models read and verify them. Keep the expected names out of model prompts.
4. **1:40–2:10, tool recovery.** Remove the cap, press **Break the next tool
   call**, then reintroduce it after the tracking memory expires. Show the error
   and the attempted recovery. With a second working camera, check whether the
   Director finishes the investigation there.
5. **2:10–2:40, camera fallback.** With both cameras able to see the prop, cover
   the hero lens and introduce the fedora to scout1. Show which camera supplies
   the evidence and whether its crop supports the report. Rehearse this step
   only with a working second camera.
6. **2:40–3:00, close.** Compare frames observed with VLM calls made. “The first
   stage found a change. The vision models inspected the hat and checked the
   claim.” Use the measured timings for this run; the YOLO timing above does
   not describe the motion backend.

If time allows, repeat the cap step with the lettering turned away. A fresh
crop can support “black baseball cap”; it cannot establish hidden lettering.
Inspect both the Analyst answer and Verifier verdict before calling that a
successful uncertainty check.

### Security sequence: cup, then knife

The weapon flow, on the default YOLO Scout (`--mock --synthetic` plays the same
beats in the scripted room, which is the B-roll). Knife, scissors and baseball
bat escalate; everything else YOLO names is benign.

1. **Cup.** Hold up a cup: Scout tags it benign, the Director ignores it,
   0 VLM calls, no snapshot.
2. **Knife (or scissors).** The camera zooms on the weapon, the Analyst
   describes it, the camera zooms on the head for a face crop that no model
   ever sees, the Verifier confirms the weapon itself on the weapon crop, and
   the dashboard shows the alert with both crops. Point at latency per stage.
3. **Recovery.** Press **Break the next tool call** or **Corrupt next tool
   args**: the Director gets an error, recovers, still reports.
4. **Fallback.** Cover the hero lens: the investigation finishes on scout1.
5. **Close** on the ratio: frames handled by a 10 ms model vs. VLM calls made.
