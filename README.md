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

## Fine-tune the Scout (about 20 minutes on this Mac)
Stock YOLOv8n knows COCO's 80 classes. It has no badge class and it was never shown our room.
The fine-tune keeps the same 3.2M-parameter nano model (so the Scout stays ~10 ms a frame) and teaches it
four classes: `person`, `weapon` (knife, scissors, baseball bat), `badge`, `bottle`.

**Open data.** Most of the training set is an open COCO 2017 subset: COCO's own boxes for knife, scissors,
baseball bat, bottle and person, kept only where the image licence allows CC-BY redistribution
(Attribution, No known copyright restrictions, US Government Work). That is 1,279 images
(1,789 weapon, 1,218 bottle and 4,399 person boxes), about 16% of the COCO images with those objects.
It puts the hackathon's "must use open data" rule inside the model the demo actually runs, with
per-image licence and source URL in `data/coco_subset/manifest.jsonl`. COCO has no firearms and we don't fake any.

Everything runs on the laptop with the camera attached; no Brev needed.
```bash
# open COCO subset (~30 s measured: annotations + 1,279 images)
python scripts/build_scout_dataset.py coco
# shoot each prop; repeat with --prop bat, scissors, badge, bottle,
# then --prop none (empty room, empty hands)
python scripts/capture_props.py --camera hero --prop knife --n 40
# stock YOLOv8 drafts the boxes; fix the held-out test photos
# first (they grade the eval), then the rest
python scripts/build_scout_dataset.py prelabel
python scripts/review_labels.py --split test
python scripts/review_labels.py
python scripts/build_scout_dataset.py build
# train (cuda > mps > cpu, auto) and score before vs after
python scripts/train_scout.py
python scripts/eval_scout.py --after runs/scout/weights/best.pt
YOLO_MODEL=runs/scout/weights/best.pt python run.py
```
Aim for 100-200 prop photos in total. Move the prop, switch hands, change distance and angle while
`capture_props.py` runs. In-room photos are split by time within each session (first 70% train, last 20%
test), so near-identical neighbouring frames never land on both sides. `build` refuses to run while
test photos still carry unreviewed drafts: otherwise the eval would grade stock YOLO against its own guesses.

**Training time, measured on this M5 Max (MPS).** YOLOv8n at 640 px, batch 32, on the 1,135
open COCO train+val images: about 29 s per epoch once warm (24-25 s training, 4 s validation;
the first epoch takes 44 s), 3 epochs in 117 s wall time. At ~1,500 images that is about 35 s
per epoch, so the MPS default of 30 epochs takes about 19 minutes and `--epochs 20` about 12.
mAP50 on the COCO validation images was already 0.45 after 3 epochs. On a CUDA box
`train_scout.py` switches to 60 epochs at batch 64 by itself; copy `data/scout/` over first.

`eval_scout.py` prints precision and recall per class, before (stock YOLOv8n, COCO names mapped) vs after,
on the held-out test photos, at the 0.35 confidence the runtime Scout uses. Its last line is the number for
the video: `Scout weapon detection (room test photos, conf 0.35): recall X% -> Y%, precision ...`.
Routing comes from the weapon flow in `agents/scout.py`: `weapon` escalates, `badge` and `bottle` are benign
(ignored, no VLM call), `person` is context for the face crop.

**Publishing (owner approval first).** `python scripts/publish_scout_hf.py --model-repo ORG/sherlock-scout
--dataset-repo ORG/sherlock-scout-data` stages both repos with cards under `runs/hf_staging/` and lists the
files; it uploads nothing without `--publish` (and then only to private repos unless `--public`). The
dataset card is CC-BY-4.0. The weights card defaults to AGPL-3.0 because they are fine-tuned from
Ultralytics YOLOv8n (AGPL-3.0); `--model-license` overrides that. In-room photos show real people, so they
stay out of the upload unless `--include-room-photos`, and that needs everyone pictured to agree.

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
scripts/             ptz_probe.py, plus the Scout fine-tune kit: capture_props, build_scout_dataset,
                     review_labels, train_scout, eval_scout, publish_scout_hf (shared: scout_data.py)
tests/               End-to-end mock run: alert chain, ignore path (zero VLM calls), evidence, recovery;
                     fine-tune kit incl. a tiny CPU train + eval
```

## Development
The lint gate is one command, and it must report zero findings:
```bash
uvx ruff@0.17.1 check .
```
`ruff.toml` pins this exact version (`required-version`; other versions refuse to run) and spells out its rule set, so every machine gets the same answer.

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
