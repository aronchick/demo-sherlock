# Sherlock — a camera that investigates

Small models that notice, decide, zoom, read, and verify. NVIDIA × Red Hat Small Models Hack, Raleigh, Oct 2026.
Tracks: Multi-Agent (primary), Tool-Calling, Nemotron + vLLM bounties.

```
Scout (YOLOv8n detector, every frame, ~10 ms)
  └─ new object? ──► Director (Nemotron 3.5 Lightning 30B-A3B + camera tools)
                        ├─ look_at / move  → PTZ camera (Insta360 Link 2 or software crop)
                        ├─ snapshot        → Analyst (SmolVLM-500M on the zoomed crop only)
                        └─ report          → Verifier (same SmolVLM-500M instance: does the crop support the claim?)
```
One tiny model gates the expensive ones. The dashboard counts "VLM calls avoided" live.

## Architecture, end to end

Sherlock runs on two machines. The laptop with the cameras watches every
frame and decides when a model is worth calling. A Brev GPU box serves the
models with vLLM. Expanso Edge is meant to move everything that happens
after a decision (events, incidents, evidence, serving metrics, training
examples) from the laptop to a cloud core. That data plane is designed in
[Expanso Edge data plane](#expanso-edge-data-plane-proposed-not-built) but
**is not built yet**.

```
 LAPTOP (Mac, cameras attached)            BREV GPU BOX (1x H100 80GB)
 ──────────────────────────────            ───────────────────────────────
 hero:   Insta360 Link 2 (gimbal)          vLLM v0.28.0, two containers
 scout1: MacBook camera
      │ every frame (~30 ms loop)          :8001 "director"
      ▼                                      Nemotron 3.5 Lightning
 Scout: YOLOv8n, ~10 ms a frame              30B-A3B NVFP4, tool calling,
      │ new object held 0.6 s                thinking off, prefix caching
      ▼
 Orchestrator                              :8002 "vlm"
      │                                      SmolVLM-500M-Instruct,
      ▼                                      shared Analyst + Verifier
 Director tool loop ── HTTP /v1 ─────────► :8001
   look_at / move ──► gimbal or crop
   snapshot / verify ── crop JPEG ───────► :8002
   report
      │
      ▼
 snapshots/<incident>/
 dashboard on 127.0.0.1:8000
```

### One detection, step by step

1. **Frames to the Scout.** The orchestrator loop
   (`sherlock/sherlock/orchestrator.py`) reads a wide frame from each camera
   about every 30 ms and passes it to the Scout. A failed camera is retried
   after 0.5 s without slowing the others.
2. **The Scout gates everything.** YOLOv8n (3.2M parameters, about 10 ms a
   frame) runs on the laptop (`agents/scout.py`). Its tracker raises an
   object only after it stays **0.6 s** (`DWELL_SECONDS`) and forgets it
   after **3 s**. Labels map to a threat: `weapon` (knife, scissors,
   baseball bat) escalates; `badge`, `bottle`, cup, phone and other COCO
   objects are `benign`; `person` only supplies the head box. `motion` and
   scripted-scene backends replace YOLO for the hat demo and mock mode.
3. **Duplicates and overlaps are dropped.** The same label seen on another
   camera moments ago counts as a duplicate. A detection that arrives while
   an investigation is running is counted as `dropped_busy`, not "avoided".
4. **Benign objects stop here.** The Director calls `report(ignore)` with no
   `look_at` and no `snapshot`, so **no VLM call is made**. This is the
   "VLM calls avoided" number.
5. **The Director (Nemotron) drives the camera.** `agents/director.py`
   sends the detection as JSON with the six tool schemas in
   `sherlock/sherlock/tools.py` (`look_at`, `move`, `snapshot`, `verify`,
   `list_cameras`, `report`), at temperature 0.1, 300 output tokens and
   thinking off. A weapon runs `look_at` weapon → `snapshot(weapon)` →
   `look_at` head → `snapshot(face)` → `verify` → `report(alert)`. The
   budget is `MAX_TOOL_CALLS` (6) plus 4 for weapons, to leave room for
   recovery. A tool error makes the Director try another camera or other
   arguments; prose instead of a tool call gets one nudge.
6. **`look_at` moves real hardware.** On the Insta360 Link 2 it sends UVC
   pan/tilt/zoom over USB with `link-ctl` (macOS only). Other cameras use a
   software crop (`FakePTZ`).
7. **The VLM sees only the crop.** The Analyst describes the zoomed object
   crop. The Verifier asks a yes/no question about the claim and decides on
   `log P(yes) - log P(no)` (margin 0.0 by default), or on the reply's first
   word when the server returns no usable logprobs
   ([Verifier verdicts](#verifier-verdicts)).
8. **The face crop never reaches a model.** It is saved for a human
   reviewer. Nothing identifies or describes the person.
9. **The orchestrator has the last word.** It re-checks the final claim
   against the current weapon crop, and an `alert` without a verified claim
   is **downgraded to `log`**. Evidence lands in `snapshots/<incident>/`;
   every step streams to the dashboard over SSE.
10. **Fallbacks.** If the Director endpoint is unreachable, `RuleDirector`
    runs the same tools with no LLM. A covered or failed hero camera hands
    the investigation to `scout1`. `--mock --synthetic` runs the whole loop
    with no models and no cameras.

### Where each piece runs

| Piece | Runs on | Code | What it is |
|---|---|---|---|
| Scout | Laptop | `agents/scout.py` | YOLOv8n, stock or fine-tuned here |
| Orchestrator, tools, cameras | Laptop | `orchestrator.py`, `tools.py`, `camera/` | Loop, tool dispatcher, gimbal control |
| Director | Brev `:8001`, vLLM | `agents/director.py` | Nemotron 3.5 Lightning 30B-A3B NVFP4 |
| Analyst + Verifier | Brev `:8002`, vLLM | `agents/vision.py` | One shared SmolVLM-500M-Instruct |
| `RuleDirector` | Laptop | `agents/director.py` | No-LLM fallback with the same tools |
| Dashboard | Laptop, `127.0.0.1:8000` | `dashboard/` | FastAPI, SSE events, MJPEG feeds |
| vLLM benchmark | Laptop, calling Brev | `scripts/bench_vllm.py` | Latency, tool validity, prefix-cache A/B |
| NeMo Gym ablation | Laptop, calling Brev | `gym/` | Director policy evaluation |
| Scout fine-tune | Laptop (MPS) or any CUDA box | `scripts/*scout*` | The only training in the project |

**Nemotron** is the Director only:
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (30B total, 3B
active), started by `brev/serve.sh` with `--tool-call-parser qwen3_coder
--reasoning-parser nemotron_v3`. Configuration B swaps in Nemotron Nano
9B/12B v2 and Llama-3.1-Nemotron-Nano-VL-8B.

**vLLM** serves every generative model. `brev/serve.sh` starts two
`vllm/vllm-openai:v0.28.0` containers on one GPU: the Director with 0.60 of
GPU memory, 32,768 tokens of context and 16 sequences; the VLM with 0.30,
8,192 and 8, one image per prompt. Prefix caching is the optimization under
test (`PREFIX_CACHING=1` or `0`). No speculative-decoding draft model is
loaded, which is what keeps Configuration A under the 33B cap. The laptop
reaches both through `DIRECTOR_BASE_URL` and `ANALYST_BASE_URL`, directly or
over an SSH tunnel. Model calls are plain OpenAI-compatible HTTP, so a local
Ollama or mlx-vlm server is a one-line swap for rehearsal (not bounty
evidence: the bounty requires vLLM).

**Fine-tuning** touches only the Scout. Nemotron and SmolVLM run stock
weights. The Scout kit builds an open COCO 2017 CC-BY subset plus prop
photos from the room, trains YOLOv8n on four classes in about 20 minutes on
the M5 Max, and scores before vs after
([Fine-tune the Scout](#fine-tune-the-scout-about-20-minutes-on-this-mac)).
`YOLO_MODEL=runs/scout/weights/best.pt` loads the result.

**NeMo Gym** evaluates; it does not train. Eight scripted tasks and a fixed
verifier score the Director while YAML configs change one component at a
time: prompt, tool descriptions, thinking, or Director model
([NeMo Gym ablation](#nemo-gym-ablation)). No weights change.

### Expanso Edge data plane (proposed, not built)

**Status today:** nothing in this repository runs on Expanso. The dashboard
footer that mentions Expanso shows "not connected" unless
`SHERLOCK_EXPANSO=connected` is set by hand. The design below is what the
footer should describe once it is real.

**What Expanso carries, and what it does not.** Model calls stay direct
HTTP from the orchestrator to vLLM: they sit inside a tool loop where every
millisecond shows on stage. Expanso Edge moves what happens after each
decision, on the laptop and in the cloud, and Expanso Cloud deploys and
monitors every job by node label.

```
 LAPTOP: node sherlock-edge               CLOUD: node sherlock-core
 (role=sherlock-edge)                     (role=sherlock-core)

 orchestrator                             C1 sherlock-core-intake
   ├─POST─► E1 sherlock-telemetry ──┐       ├─► incidents/  + alert webhook
   ├─POST─► E2 sherlock-incidents ──┼──────►├─► telemetry/
   └─POST─► E3 sherlock-examples ───┘       └─► examples/  (next Scout set)

 dashboard ◄──────────────────────────── C2 sherlock-vllm-metrics
                                            ◄── :8001/metrics, :8002/metrics

 Expanso Cloud: deploys all five jobs by label, shows node
 health, job revisions and rollbacks
```

| Job | Node | Input | What it does | Output |
|---|---|---|---|---|
| E1 `sherlock-telemetry` | Laptop | `http_server` on loopback; the orchestrator POSTs each dashboard event | Drops image fields; adds node, camera, model and Git commit; batches | Core intake `/telemetry` |
| E2 `sherlock-incidents` | Laptop | `http_server`; one POST per report | `switch` on action: `alert` goes out at once with the weapon crop; `log` and `ignore` send metadata only. The face crop is replaced by its SHA-256 and local path | Core intake `/incident` |
| E3 `sherlock-examples` | Laptop | `http_server`; object crops the Verifier rejected or scored near the threshold | Hard examples for the next Scout fine-tune. Object crops only, never faces. Off by default | Core intake `/example` |
| C1 `sherlock-core-intake` | Cloud | `http_server /{endpoint}` | Validates, stamps receipt time, routes by endpoint; failed messages are logged and rejected, not dropped silently | `file` stores per stream; `http_client` alert webhook |
| C2 `sherlock-vllm-metrics` | Cloud (Brev box) | `generate` every 10 s, then GET both vLLM `/metrics` | Per-model KV-cache use, prefix-cache hits, queue depth, request counts | Core store and the laptop dashboard |

Every component named here (`http_server`, `http_client`, `generate`,
`switch`, `file`, Bloblang `.hash()`) is in the Expanso Edge
documentation; the job layout follows `demo-edge-to-core-fleet`
(`core-intake`) and `demo-edge-zoom-on-interest`.

**Code changes to build it:**

1. A small event sink in `sherlock/sherlock/events.py` that POSTs events and
   reports to E1 and E2 when `SHERLOCK_EDGE_URL` is set, and does nothing
   when it is not, so `just up` keeps working with no Expanso.
2. `jobs/` with the five job files, deployed through Expanso Cloud.
3. Dashboard: read C2's metrics, and drive the Expanso footer from a real
   node heartbeat instead of the `SHERLOCK_EXPANSO` flag.
4. Expanso credentials and node identity in project-local, gitignored paths
   with owner-only permissions; never a global profile.

**Open decisions:**

1. **Where the core node runs.** The Brev box already exists and sits next
   to vLLM, but stopping Brev stops the core. A small persistent cloud VM
   keeps incidents and history between sessions.
2. **Whether the face crop ever leaves the laptop.** Proposed: no. Only its
   hash and local path go up, so a reviewer opens it on the device.
3. **Whether E3 starts on.** Proposed: off. Room photos show real people,
   and the Scout publisher already requires everyone pictured to agree.

### Not measured yet

- **vLLM on the H100:** latency, tool-call validity and the prefix-cache
  A/B have only run against the local stub
  ([Results pending live endpoint](#results-pending-live-endpoint)).
- **NeMo Gym ablations:** live results are pending Brev endpoints.
- **Verifier threshold:** fitted on 50 COCO crops, not on room photos.
  SmolVLM-500M confirmed 27/30 real weapon crops and accepted 1/20 false
  "kitchen knife" claims.
- **Linux gimbal control:** the Link 2 cannot be aimed on Linux; the hero
  comes up offline there and the Director uses the next camera.

## Model lineup

### Configuration A (default: 30.51B generative parameters by model card, 32.09B by checkpoint tensor count)
Every generative model runs on vLLM (`vllm/vllm-openai:v0.28.0`), covering both the NVIDIA Nemotron Lightning bounty and the Red Hat vLLM bounty. Speculative-decoding draft models (DSpark, DFlash, MTP) are disabled because draft weights count toward the parameter budget.

| Role | Hugging Face Repo ID | Verified Parameters | Serving Details |
| :--- | :--- | :--- | :--- |
| **Director** | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | **30B total / 3B active** on model card. The BF16 checkpoint's safetensors headers hold **31.58B** (`31,577,940,288`) without MTP and **32.91B** (`32,913,266,240`) with its 270 embedded `mtp.*` speculative draft tensors (`1,335,325,952`), which vLLM does not load when speculative decoding is off ([evidence](#parameter-budget-and-speculation)) | vLLM on `:8001`, `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3`, thinking OFF by default (`DIRECTOR_THINKING=0` passes `chat_template_kwargs: {"enable_thinking": false}` for vLLM and `reasoning_effort: "none"`, which Ollama needs and vLLM v0.28.0 maps to the same switch) |
| **Analyst + Verifier** | `HuggingFaceTB/SmolVLM-500M-Instruct` | **0.51B** (`507,482,304` safetensors parameters; `Idefics3ForConditionalGeneration` in vLLM v0.28.0) | Single shared vLLM instance on `:8002` (counted once). `HuggingFaceTB/SmolVLM-Instruct` (2.25B) and `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (2.25B) also run on vLLM v0.28.0 but put Configuration A at 33.82B by tensor count, over the cap |
| **Scout** | `yolov8n.pt` (Ultralytics YOLOv8n) | **3.2M** (excluded) | Non-generative detector; excluded from the generative parameter total |
| **Total (generative)** | | **30.51B** by model card (30B + 0.51B); **32.09B** by tensor count without MTP (31.58B + 0.51B, **0.91B under the 33B cap**); 33.42B counting the never-loaded MTP tensors | Zero draft models |

Configuration A is under the 33B cap by model card and by checkpoint tensor count. Only counting the 1.34B of MTP tensors that vLLM never loads takes it to 33.42B, over the cap; the evidence that they are not loaded is in [Parameter budget and speculation](#parameter-budget-and-speculation). Under the tensor count, the shared VLM must be at most **1.42B** (33B − 31.58B).

*Why SmolVLM-500M instead of Moondream 2 (1.9B):* `vikhyatk/moondream2` (`1,927,237,104` parameters, `HfMoondream` / `moondream1`) is not in vLLM v0.28.0's supported models list (`vllm` v0.28.0 only implements `Moondream3ForCausalLM` for the 9B MoE `moondream/moondream3-preview`). At 1.93B it would also exceed the 1.42B tensor-count headroom. `HuggingFaceTB/SmolVLM-500M-Instruct` runs natively in vLLM v0.28.0 via `Idefics3ForConditionalGeneration`; vLLM's own model tests use its 256M sibling, which has the same architecture and vision encoder, as the small `Idefics3` example.

### Configuration B (NeMo Gym ablation, ~17.6B–21.0B generative parameters)
Configuration B swaps the Director and shared VLM via `.env` (see the commented-out Configuration B block in `.env.example`) for the NeMo Gym ablation:
- **Director**: `nvidia/NVIDIA-Nemotron-Nano-9B-v2` (**8.89B**, `8,888,227,328` safetensors parameters) or `nvidia/NVIDIA-Nemotron-Nano-12B-v2` (**12.31B**, `12,310,001,152` safetensors parameters).
- **Analyst + Verifier (shared)**: `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` (**8.72B**, `8,721,768,454` safetensors parameters; Llama-3.1-8B-Instruct + C-RADIOv2-H vision encoder, supported in vLLM v0.28.0 via `Llama_Nemotron_Nano_VL`).
- **Total generative parameters**: **17.61B** (with Nano 9B v2) or **21.03B** (with Nano 12B v2).
- **Tradeoff**: Trades the Nemotron 3.5 Lightning bounty for a larger 8.72B VLM on zoomed crops while staying well under budget. Do not combine Nemotron 3.5 Lightning (30B) with a 7–8B VLM (38.72B by model card, 40.30B by tensor count; over budget either way).

## Run with nothing (sandbox, airplane, dead wifi)
```bash
just up                                    # needs uv and just; ultralytics (the `yolo` extra) is optional for this mode
open http://127.0.0.1:8000
just down                                  # stops what `just up` started
```
`just up` runs `run.py --mock --synthetic` in the background on `DASHBOARD_PORT` (default 8000) with
`CAMERAS=hero:fake:synthetic,scout1:fake:synthetic` (set either in your shell to change it), and keeps its
PID and log in the gitignored `.runtime/`. The first run builds the environment from `pyproject.toml` and `uv.lock`.

The dashboard binds to `127.0.0.1` by default. Keep demo and recording UIs local:
they serve live camera feeds and saved face crops. Set `DASHBOARD_HOST=0.0.0.0`
in the shell or `.env` **only on a trusted network** to deliberately allow LAN
access. Unset or blank `DASHBOARD_HOST` values keep the loopback default.

Rule-based director, mock VLMs, a scripted room on a 26 s loop: someone holds up a cup (ignored, 0 VLM calls),
then someone holds up a knife (look_at → snapshot → look_at head → face snapshot → verify → alert with both crops).
Press **Break the next tool call** or **Cover hero lens** on the dashboard to see recovery; corrupted tool arguments have
no button, so send `curl -X POST 127.0.0.1:8000/demo/corrupt/1`.
Evidence lands in `snapshots/<incident>/` (`EVIDENCE_DIR` overrides). The face crop is stored for a human only:
no model ever sees it, and nothing identifies or describes the person. It is kept only when the Verifier confirms the
weapon claim; a rejected or unverified claim discards it before anything is saved.

**Data flow view (the default page).** `http://localhost:8000/` shows each investigation as it moves: the camera feeds
into the Scout, only a ticket and then a crop cross to the models, and the decision lands in Ignored, Logged
or Alert (benign objects are archived with their picture) while an incident record fills in stage by stage.
`/topology` is the simpler page for explaining what runs where: the laptop (cameras, YOLO Scout, orchestrator,
evidence) and the Brev GPU box (vLLM serving the Director and the shared Analyst/Verifier), with the configured
model names; set `BREV_HOST=<name>` when the endpoints are reached over an SSH tunnel so the page names the box.
Every page names the serving program (vLLM, Ollama, mlx-vlm) only from what each endpoint answers, and shows the
host when it cannot tell; `MODEL_SERVER` (or `DIRECTOR_`, `ANALYST_`, `VERIFIER_MODEL_SERVER`) declares it instead.
When every endpoint is on the laptop, the topology page greys out the Brev box.
The Sherlock board (the cascade, the chain and the incident card) lives at `/log`. It only reads the existing `/events` and `/stream`
endpoints and loads no network resources (its fonts are vendored under `dashboard/static/fonts/`, SIL OFL).
The footer notes that Expanso Edge processes this data and Expanso Cloud deploys and monitors it, and says
"not connected" until you set `SHERLOCK_EXPANSO=connected` for a run where an Edge pipeline really is doing so.

## vLLM on Brev (Config A default)
1. On the Brev GPU box (1x H100/A100 80GB), start both vLLM containers:
```bash
./brev/serve.sh
```
2. On the Mac, copy `.env.example` to `.env` and point at the Brev host (or local SSH tunnel on `:8001` and `:8002`):
```bash
cp .env.example .env              # CAMERAS=hero:fake:0,scout1:fake:1 opens two webcams (software PTZ); with one, set CAMERAS=hero:fake:0
just live                         # real cameras and models per .env; installs the `yolo` extra (ultralytics)
```
Open <http://127.0.0.1:8000> on the Mac; the dashboard stays local while model calls
use the Brev endpoints or SSH tunnels.
If the endpoint is unreachable mid-pitch, the Director automatically drops to the rule-based fallback (`RuleDirector`).

## Hardware: Insta360 Link 2 hero camera
The hero is an **Insta360 Link 2, the gimbal model** (USB `2e1a:4c04`), not the gimbal-less Link 2C. The scout is the Mac's built-in webcam on the software-crop backend.

1. Plug the Link 2 **straight into a Mac port, not the hub**. It is a USB 2.0 device; sharing a hub with other cameras starves the stream.
2. Quit the **Insta360 Link Controller** app, or at least turn AI Tracking off in it. Firmware tracking fights `look_at`. The app is not needed for control.
3. `uv run scripts/ptz_probe.py`: reports whether the camera is on USB and behind a hub, whether USB-direct PTZ answers, the pan/tilt/zoom ranges it reports, the AI mode, and each OpenCV index with its camera name. Run it from a terminal that has macOS camera permission.
4. `uv run scripts/ptz_probe.py --wiggle --index <n>`: moves the gimbal and writes `snapshots/probe_*.jpg`. +pan must move the view right and +tilt up; otherwise set `INSTA360_PAN_SIGN=-1` / `INSTA360_TILT_SIGN=-1`.
5. Select cameras **by name**, not index:
   ```bash
   CAMERAS="hero:insta360:name:Insta360 Link 2,scout1:fake:name:MacBook Pro Camera"
   ```
   OpenCV numbers cameras by sorted AVFoundation unique ID, so indexes reshuffle whenever a device (Link 2, iPhone Continuity Camera, OBS Virtual Camera) appears or disappears, and two cameras can silently swap mid-demo. `name:` matches the names `ptz_probe.py` prints (exact, case-insensitive, or a unique substring); a camera that will not open (missing or ambiguous name, bad index) is left out with a warning in the log and on the dashboard, and Sherlock exits only if no camera opens. Numeric indexes still work.

Control path on macOS: standard UVC pan/tilt/zoom control transfers sent straight over USB with [link-ctl](https://pypi.org/project/link-ctl/)'s IOKit module (no sudo, no app). On Linux the Link 2 cannot be aimed: its stock driver reads pan/tilt but rejects setting it, and the only workaround (detaching `uvcvideo` for a raw USB transfer) drops the video stream ([link-ctl notes](https://github.com/csmarshall/link-ctl/blob/main/docs/LINK2_LINUX.md)). The v4l2 path checks this when it opens, so on Linux the hero comes up offline and the Director uses the next camera. The Link Controller WebSocket API is not used because it cannot set an absolute pan/tilt. If the gimbal does not answer within `INSTA360_CMD_TIMEOUT_S`, `look_at`/`move` return a tool error and the Director moves to the next camera.

Tuning (env, defaults in `sherlock/sherlock/camera/insta360.py`):

| Variable | Default | What it does |
| :--- | :--- | :--- |
| `INSTA360_FILL` | `0.6` | Share of the frame the target fills after `look_at` |
| `INSTA360_MAX_ZOOM` | `4.0` | Zoom cap (Link 2 zoom is 4x digital; lower it if crops get soft) |
| `INSTA360_HFOV_DEG` | `67` | Horizontal FOV at 1x, from the Insta360 spec sheet |
| `INSTA360_SETTLE_S` / `INSTA360_SLEW_DEG_S` / `INSTA360_ZOOM_SETTLE_S` | `0.35` / `120` / `0.1` | Wait before a snapshot: fixed + travel/speed + per-1x zoom change |
| `INSTA360_HOME_AFTER_S` | `6` | Return to the wide home pose after this many idle seconds (`0` = never) |
| `INSTA360_PAN_SIGN` / `INSTA360_TILT_SIGN` / `INSTA360_UNITS_PER_DEG` | `1` / `1` / `3600` | Direction and units of the raw UVC values (arc-seconds per spec) |

The OBSBOT adapter (`hero:obsbot:<n>`) is kept as an alternate backend.

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
uv run scripts/build_scout_dataset.py coco
# shoot each prop; repeat with --prop bat, scissors, badge, bottle,
# then --prop none (empty room, empty hands)
uv run scripts/capture_props.py --camera hero --prop knife --n 40
# stock YOLOv8 drafts the boxes; fix the held-out test photos
# first (they grade the eval), then the rest
uv run --extra yolo scripts/build_scout_dataset.py prelabel
uv run scripts/review_labels.py --split test
uv run scripts/review_labels.py
uv run scripts/build_scout_dataset.py build
# train (cuda > mps > cpu, auto) and score before vs after
uv run --extra yolo scripts/train_scout.py
uv run --extra yolo scripts/eval_scout.py --measured \
  --after runs/scout/weights/best.pt
YOLO_MODEL=runs/scout/weights/best.pt just live
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
Choose `--measured` only for real training and evaluation data. After
`train_scout.py --smoke`, use `eval_scout.py --smoke` instead. Smoke training
defaults to `runs/scout_smoke/weights/best.pt`, preserving the
normal `runs/scout/weights/best.pt` and its measured evaluation. Pass the smoke
weights with `--after` when checking that run. An explicit training `--name`
overrides the default directory.
Evaluation requires an explicit mode; an invocation without it stops before
evaluation. Measured output defaults to
`runs/scout/eval.json`; smoke output goes to `runs/scout/eval_smoke.json` so it
does not replace measured results. The dashboard excludes smoke reports, and
the publisher rejects smoke or unlabelled evaluation files before staging.
Routing comes from the weapon flow in `sherlock/sherlock/agents/scout.py`: `weapon` escalates, `badge` and `bottle` are benign
(ignored, no VLM call), `person` is context for the face crop.

**Publishing (owner approval first).** `uv run --with huggingface-hub scripts/publish_scout_hf.py --model-repo ORG/sherlock-scout
--dataset-repo ORG/sherlock-scout-data` stages both repos with cards under `runs/hf_staging/` and lists the
files; it uploads nothing without `--publish` (and then only to private repos unless `--public`). The
dataset card is CC-BY-4.0. The weights card defaults to AGPL-3.0 because they are fine-tuned from
Ultralytics YOLOv8n (AGPL-3.0); `--model-license` overrides that. In-room photos show real people, so they
stay out of the upload unless `--include-room-photos`, and that needs everyone pictured to agree.

## vLLM

vLLM is the serving layer for Sherlock's investigation loop. The Director
chooses camera tools; the Analyst reads a zoomed crop; the Verifier checks
the resulting claim against that crop. Scout gates these calls locally.
The planned Brev setup runs two vLLM **0.28.0** containers on one H100:

| Role | Model | Served name | Port |
|---|---|---|---:|
| Director | Nemotron 3.5 Lightning 30B, NVFP4 repository | `director` | 8001 |
| Analyst + Verifier | `HuggingFaceTB/SmolVLM-500M-Instruct`, 0.51B, shared | `vlm` | 8002 |

The Director uses automatic tool selection and turns thinking off in every
request: `chat_template_kwargs.enable_thinking: false` for vLLM, plus
`reasoning_effort: "none"`, which Ollama needs (without it, a local
rehearsal spent 2.4 to 5.0 s reasoning on the first turn; with it, 0.35 to
0.40 s). Analyst and Verifier share one VLM to avoid
loading a third model. Separate context, sequence and memory budgets let
both servers share the GPU. The initial `brev/serve.sh` budgets are 32,768
tokens / 16 sequences / 0.60 GPU memory for Director and 8,192 / 8 / 0.30
for VLM. These are starting settings, not measured capacity; the serving
owner must confirm model revisions, parser, startup flags and available
memory on the H100. This benchmark does not change `serve.sh`.

**Live H100 measurements are pending.** The benchmark is validated against
a local streaming stub. Stub results test the measurement code and must
not be submitted as vLLM performance evidence.

### Verifier verdicts

The Verifier asks the VLM a yes/no question about the weapon crop:

> Is the object in this image {claim}? Judge the object only; do not
> identify or describe any person. Answer yes or no.

An earlier version asked for a JSON verdict. In a local bake-off on 50 open
COCO crops (a true and a false claim each), the SmolVLM models ignored that
instruction (`True.`, `Supports.`) and leaned towards yes even when the reply
was constrained to the JSON schema. They answer the yes/no question
directly: every reply from the three models tested began with "Yes" or "No".

| Verifier, accuracy /100 | SmolVLM-500M | SmolVLM-Instruct (2.2B) |
|---|---:|---:|
| JSON schema + tuned log-odds threshold (leave-one-crop-out) | 74 | 95 |
| yes/no, first word of the reply | 94 | 98 |
| yes/no + tuned log-odds threshold (leave-one-crop-out) | 95 | 99 |

How `Verifier.check` decides:

- **With logprobs** (vLLM, llama.cpp's server), it requests `top_logprobs`
  and computes `log P(yes) - log P(no)` at the reply's first word, pooling
  spellings (` Yes`, `yes`, ...). The claim is supported when this margin
  exceeds `VERIFIER_MARGIN`, default **0.0** (whichever answer is likelier).
  Fitted on the bake-off: **-2.75** for SmolVLM-500M and **-1.88** for
  SmolVLM-Instruct, because the models lean "no" under this question.
  Retune on room photos before using them.
- **Without logprobs** (the field refused with HTTP 400/422/501, which is
  remembered for the run, or ignored, as by some Ollama and MLX servers), it
  reads the reply's first word. A reply that opens with neither yes nor no
  is not support. At the default threshold this read agreed with the margin
  on all 100 bake-off calls for each model.

Each verdict records its mode (`margin` or `text`) in the dashboard event
and the console line. SmolVLM-500M at the default threshold confirmed 27/30
real weapon crops, accepted 1/20 false "kitchen knife" claims on cups and
bottles, and accepted 0/30 wrong-weapon claims on weapon crops.

The prompt changed, so the benchmark's workload hash, which records
`Verifier.PROMPT`, changes too. The NeMo Gym schemas do not: the verdict
still carries `supported`, `confidence` (`null` without logprobs) and
`reason` (the model's reply), plus `mode` and `margin`.

### Parameter budget and speculation

Configuration A's model-card budget is approximately **30.51B** generative
parameters: the 30B Director plus the shared 0.51B VLM, counted once.
The checkpoint tensor count is higher. Summing tensor shapes in the 14
safetensors shard headers of
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` gives:

| Count | Director | + `SmolVLM-500M-Instruct` (`507,482,304`) |
|---|---:|---:|
| Model card | 30B | 30.51B, under the cap |
| Tensors without MTP | `31,577,940,288` (31.58B) | **32.09B, 0.91B under the cap** |
| Tensors with 270 `mtp.*` (`1,335,325,952`) | `32,913,266,240` (32.91B) | 33.42B, over the cap if never-loaded MTP is counted |

Hugging Face's model page rounds the BF16 repository to about 31.6B. Its
API total (`31,577,937,344`) also leaves out MTP and sits 2,944 below the
header count, exactly the base model's 23 FP32 router-bias tensors of 128
values each. The NVFP4 repository that `serve.sh` runs packs two FP4
weights per `U8` byte, so its raw element count (19.71B) is not a parameter
count; the BF16 checkpoint of the same model is the reference. Under the
tensor count, the shared VLM must be at most 1.42B (33B − 31.58B) to stay
under the cap.

Speculative draft weights count toward the 33B cap. `brev/serve.sh` passes
no `--speculative-config` / `-sc`, `--spec-method`, `--spec-model`,
`--spec-tokens`, or other MTP/draft settings.

The [vLLM v0.28.0 engine arguments](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/engine/arg_utils.py)
default `speculative_config` to `None`; `create_speculative_config` returns
`None` when no speculation options are supplied. The
[automatic speculator detection](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/transformers_utils/config.py)
only overrides this when the checkpoint has `speculators_config`. The
[Lightning NVFP4 configuration](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/raw/main/config.json)
inspected for this build has no such key and selects `NemotronHForCausalLM`.

Lightning includes 270 embedded MTP tensors (1.34B parameters), but the
[v0.28.0 NemotronH base-model loader](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/nemotron_h.py)
uses `AutoWeightsLoader(self, skip_prefixes=["mtp"])`. Those weights are
skipped when loading the base model; their presence in the checkpoint does
not enable MTP speculation. The Director therefore loads the 31.58B count,
not 32.91B. This is configuration and source-code evidence.
Live container startup and its effective model configuration still need
verification on the H100.

### Run the benchmark

To check the complete flow without a GPU or model endpoint:

```bash
uv sync
just bench-stub
just preflight-stub
```

`bench-stub` runs both workloads at concurrency 1, 4, 8 and 16.
`preflight-stub` checks model readiness, sends the Director tool-call and
JPEG smoke requests, then runs both sweeps and the A/B comparison. The
cache-off/cache-on sequence is simulated: it never starts or stops Docker
containers and demonstrates no caching benefit. The stub derives tool
arguments from the request's current schemas; it performs no inference.

Both commands start their own loopback-only server, override configured
model endpoints for that run, and stop the server before returning. They
force `evidence_kind: stub` and `execution_mode: local_stub`. Token counts
and cache metrics are synthetic; timings measure only local protocol work.
Outputs default to the ignored `.runtime/vllm-stub/`, outside the dashboard's
`results/` directory. `--output-dir` can choose another directory, but
`results/` and symlinks into it are rejected in stub mode. The dashboard
also rejects the stub evidence marker if a report is copied there.

The same entry points are available directly:

```bash
uv run scripts/bench_vllm.py --stub --metrics
uv run scripts/brev_preflight.py --stub --mode ab --requests 16
```

The preflight recipe uses 16 requests per cell for a quick protocol check;
pass `--requests 200` to either recipe to exercise larger result tables.

Set `DIRECTOR_BASE_URL` and `ANALYST_BASE_URL` to the two OpenAI-compatible
base URLs, including `/v1`. With those set, this is the complete command:

```bash
uv run scripts/bench_vllm.py --metrics
```

`uv` installs the script's declared dependencies. Served model names default
to `director` and `vlm`; override with `DIRECTOR_MODEL` / `ANALYST_MODEL`.
Use `DIRECTOR_API_KEY` / `ANALYST_API_KEY` when required, with
`OPENAI_API_KEY` as fallback. Keep secrets out of commands and results.
`--help` lists the other options. Endpoint URLs and response text are not
saved in the results.

The command tests concurrency **1, 4, 8, 16**, with 64 measured requests per
cell and four excluded warmups. Director traffic imports `SYSTEM` from
`sherlock/sherlock/agents/director.py` and `TOOL_SCHEMAS` from `sherlock/sherlock/tools.py`.
It varies weapon, unknown and benign detections, including `threat` and
`head_bbox`. Weapon traffic samples each step through verification and the
final report, with and without face-evidence steps. Benign traffic has no
prior camera calls. Weapon prompts add the application's four-call recovery
budget. Fixture arguments are checked against the imported schemas before
dispatch; older schemas omit unsupported fields and steps. Requests use
the application's temperature and token limits.
It does not execute camera tools. The shared prefix is the real prompt and
tool schemas; no padding is added to inflate a caching benefit.

VLM traffic alternates the imported Analyst and Verifier prompts using a
[real JPEG object crop](tests/fixtures/vllm/README.md) of a model spacecraft,
with no face in the image. Director face-tool results are text metadata;
face crops are never sent to the VLM. Supply a representative object
crop with `--image path/to/crop.jpg` for the final run. This fixed-image
test includes repeated image/prompt cache reuse; it does not represent a
stream of unique scenes. Workloads run sequentially, so results describe
each server with both models resident, not simultaneous Director/VLM load.

Each run writes `results/vllm_bench_<UTC timestamp>.json` and a matching
Markdown table, and prints the table. JSON `schema_version: 1` contains:

- `evidence_kind` (`endpoint` or `stub`), operator-declared `environment`
  and `prefix_caching`, Git commit, settings, workload hash and JPEG hashes.
- `runs[]`, keyed by `workload`, `model`, `concurrency`, with `summary`,
  excluded `warmup` failure counts, optional `metrics`, and per-request
  `samples`. A dashboard can read these files without scraping Markdown.
  Filter to `evidence_kind: endpoint` before displaying model performance;
  the checked-in `stub` example is only a protocol smoke test.
- `summary.ttft_ms` and `e2e_ms` with `p50`, `p95`, `p99`,
  `output_tokens_per_s`, `requests_per_s`, and `tool_call_validity_rate`.

TTFT starts at HTTP dispatch and ends at the first text or tool name/argument
delta, excluding empty role/ID events. It includes network delay, server
queueing and tool-parser buffering. Latency percentiles cover successful,
complete streams. Output tokens/s is server-reported successful completion
tokens divided by cell wall time; requests/s uses successful requests over
the same interval. Missing token usage stays `null`, never estimated from
text or SSE chunk counts. No automatic retries hide failures. A timeout,
HTTP error, incomplete stream or token-limit finish is counted as failure;
the command writes the evidence and exits nonzero.

Director validity is the fraction of **all attempted requests** that return
at least one complete, named tool call with JSON arguments matching the
imported schema. Every call in that response must pass, including required
fields, types, enums, bounds and no unknown arguments. This tests call
structure, not whether the model chose the right investigation action.
Individual-call validity is also saved separately.

With `--metrics`, optional `/metrics` snapshots record KV cache usage and
prefix-cache counter deltas, excluding warmup. Missing metrics stay
unavailable. KV snapshots are not peak memory usage. A reverse proxy can
override the scrape path with `DIRECTOR_METRICS_URL` / `ANALYST_METRICS_URL`.
If metrics use the repository ID instead of the requested served alias,
the benchmark matches a single labelled model and records that choice in
`matched_model_name` / `model_match`. It leaves ambiguous models unmatched.
Avoid unrelated endpoint traffic during measurement because counters are
server-wide.

### Prefix-cache A/B

The chosen optimization is automatic prefix caching: Director requests
repeat the same system prompt and tool definitions. The
[vLLM 0.28.0 CLI](https://docs.vllm.ai/en/v0.28.0/cli/serve/#cacheconfig)
documents `--enable-prefix-caching`, `--no-enable-prefix-caching`, and
`--prefix-caching-hash-algo sha256`. The
[APC guide](https://docs.vllm.ai/en/v0.28.0/features/automatic_prefix_caching/)
explains that reuse reduces prefill work, not decoding work. The expected
signal is lower TTFT; the size of any benefit remains unmeasured.

1. On the Brev host, the serving owner starts the baseline with:

   ```bash
   PREFIX_CACHING=0 bash brev/serve.sh
   ```

   `PREFIX_CACHING=0` selects `--no-enable-prefix-caching` for Director.
   The script creates both containers; the serving owner must stop/remove
   previous experiment containers before each launch. Keep the model
   revision, parser, context, sequence/memory budgets, VLM settings and
   hardware fixed. Wait for both models to be ready.
2. Record the baseline. Use a nonsecret environment label identifying the
   GPU and exact model/config revisions; reuse it for the candidate:

   ```bash
   uv run scripts/bench_vllm.py --metrics --requests 200 \
     --prefix-caching off --environment h100-config-1
   ```

3. On the Brev host, the serving owner recreates the containers with:

   ```bash
   PREFIX_CACHING=1 bash brev/serve.sh
   ```

   This selects `--enable-prefix-caching --prefix-caching-hash-algo sha256`
   for Director and preserves the VLM settings. Wait for readiness again.
   The benchmark's `--prefix-caching` only labels a run; set it to `off`
   for `PREFIX_CACHING=0` and `on` for `PREFIX_CACHING=1`.
4. Run the candidate against the printed baseline JSON path:

   ```bash
   uv run scripts/bench_vllm.py --metrics --requests 200 \
     --prefix-caching on --environment h100-config-1 \
     --compare results/vllm_bench_<baseline_timestamp>.json
   ```

The candidate JSON and Markdown include percent changes against baseline.
Comparison rejects different workload hashes, settings, evidence types or models.
The hash includes the runtime Verifier prompt: rerun both A/B arms after a
prompt change. The numbers panel shows a cache delta only when the stored
baseline and candidate hashes both match the run; older comparisons without
these hashes show TBD for the delta.
Use only Director rows to attribute improvement to this change; VLM rows
are a control with unchanged server settings. Save the actual server flags,
container image digest and model revisions alongside final submission
evidence. Repeat the experiment in reverse order to check warmup/load drift.
Use at least 200 requests per cell; p99 from the default 64 is exploratory.
For Nemotron's hybrid layers, confirm startup's effective cache mode with
the serving owner; the 0.28.0 CLI documents `align` when prefix caching is
enabled and `none` when disabled. Do not add other optimizations to this A/B.

### Results pending live endpoint

| Workload | Concurrency | TTFT p50/p95/p99 ms | E2E p50/p95/p99 ms | Output tok/s | Req/s | Tool validity |
|---|---:|---|---|---|---|---|
| Director | 1 | pending | pending | pending | pending | pending |
| Director | 4 | pending | pending | pending | pending | pending |
| Director | 8 | pending | pending | pending | pending | pending |
| Director | 16 | pending | pending | pending | pending | pending |
| VLM | 1 | pending | pending | pending | pending | n/a |
| VLM | 4 | pending | pending | pending | pending | n/a |
| VLM | 8 | pending | pending | pending | pending | n/a |
| VLM | 16 | pending | pending | pending | pending | n/a |

Prefix-cache on/off TTFT change: **pending**. Cache hit rate: **pending**.
KV usage: **pending**. Failure rates: **pending**.
Replace these placeholders with the real endpoint tables before submitting
the Red Hat vLLM bounty explanation.

### Local validation

```bash
just lint     # uv run ruff check .  (ruff 0.17.1, pinned)
just test     # uv run pytest
```

No extra flags are needed (see Development): `pyproject.toml` installs the
nested `sherlock/sherlock` package and the `dev` dependency group, and the
root `pytest.ini` supplies the test directory.

The tests start a tiny loopback OpenAI-compatible SSE server, run the
benchmark CLI through both A/B legs and all concurrency levels, verify JSON
and Markdown, stop the server and rebind its port to prove cleanup. They
also check HTTP errors, timeouts, truncated streams, missing usage,
invalid tool calls and counter resets. No Brev endpoint is needed.

## Layout
The `sherlock/...` entries below live in the package directory, `sherlock/sherlock/`.
```
sherlock/camera/     Camera interface, FakePTZ (crop/zoom), Insta360Camera (Link 2 gimbal), ObsbotCamera (UVC), SyntheticSource
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
just lint        # uv run ruff check .
just check       # lint, then the test suite
```
`ruff.toml` pins this exact version (`required-version`; other versions refuse to run) and spells out its rule set, so every machine gets the same answer.

The test gate runs from a clean environment. `uv sync` builds it from `pyproject.toml` and `uv.lock`: the app, benchmark and test-only dependencies (the `dev` group mirrors `requirements-dev.txt`, which stays for plain `pip` installs, except that ultralytics is the optional `yolo` extra):
```bash
uv sync --extra gym
just test        # uv run --extra gym pytest
```
Two scout fine-tune tests need ultralytics and skip without it; `uv run --extra yolo pytest` runs them.

## NeMo Gym ablation

Sherlock includes a native [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) (`nemo-gym==0.7.0`, installed via the `gym` optional extra on Python >= 3.13.14) environment and multi-metric verifier under `gym/` for single-component agent ablations. The 8 scripted tasks (`gym/resources_servers/sherlock/data/tasks.jsonl`) and the verifier (`SherlockResourcesServer.verify`) stay fixed while YAML configs swap one Director agent component at a time:

1. **System prompt (`prompt_v1` vs `nemotron_lightning`)**: `gym/prompts/system_v1.yaml` vs `gym/prompts/system_v2.yaml` (adds zero-snapshot benign filtering and the two-crop weapon + head/face protocol).
2. **Tool descriptions (`terse_tools` vs `nemotron_lightning`)**: `terse` (3-word tool strings, stripped parameter hints) vs `verbose` (`sherlock.tools.TOOL_SCHEMAS`).
3. **Reasoning mode (`thinking_on` vs `nemotron_lightning`)**: `enable_thinking: true` vs `enable_thinking: false` on `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (the thinking-off configs also send `reasoning_effort: none`, so they stay off on servers that ignore the chat-template switch).
4. **Director model (`qwen_director` vs `nemotron_lightning`)**: `Qwen/Qwen2.5-7B-Instruct` (7.62B) vs `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (30B total / 3B active).

Every configuration pairs the Director with `HuggingFaceTB/SmolVLM-500M-Instruct` (0.51B) for the Analyst/Verifier VLM. By model card that is 30.51B for the Nemotron 3.5 Lightning configs and 8.12B for Qwen 2.5 7B (`7,615,616,512` + `507,482,304`). By checkpoint tensor count the Lightning configs are 32.09B (see [Parameter budget and speculation](#parameter-budget-and-speculation)). All are under the 33B limit; the ablation runner's budget check uses the model-card figures.

### Run one ablation with one command

Each YAML config declares its target endpoint variable via `endpoint_env_var`: the 4 Nemotron 3.5 Lightning configs (`prompt_v1`, `terse_tools`, `thinking_on`, `nemotron_lightning`) read `DIRECTOR_BASE_URL`, and `qwen_director` reads `QWEN_DIRECTOR_BASE_URL`. When running `python -m gym.run_ablation --all` in live mode, any config whose declared endpoint variable is unset is skipped with a clear status message instead of failing.

```bash
# Local end-to-end validation (uses the built-in OpenAI-compatible stub when no endpoint env var is set):
uv run --extra gym python -m gym.run_ablation --config gym/configs/ablation_nemotron_lightning.yaml
uv run --extra gym python -m gym.run_ablation --all --stub
```

#### Live Brev ablation run order (keeps active generative models <= 33B on each step)

1. **Run the 4 Nemotron 3.5 Lightning configs (`prompt_v1`, `terse_tools`, `thinking_on`, `nemotron_lightning`)**:
   Start the default Lightning + SmolVLM containers (`30B + 0.51B = 30.51B` model card, `31.58B + 0.51B = 32.09B` by tensor count) on the Brev host and run `--all` with `DIRECTOR_BASE_URL` set (`qwen_director` is automatically skipped because `QWEN_DIRECTOR_BASE_URL` is unset):
   ```bash
   bash brev/serve.sh
   DIRECTOR_BASE_URL=http://127.0.0.1:8001/v1 uv run --extra gym python -m gym.run_ablation --all
   ```
2. **Swap Director to `Qwen/Qwen2.5-7B-Instruct` and run `qwen_director`**:
   Switch the `director` container on `:8001` to `Qwen/Qwen2.5-7B-Instruct` (`vllm/vllm-openai:v0.28.0` with `--enable-auto-tool-choice --tool-call-parser hermes`). Replacing the `director` container instead of running both Directors simultaneously keeps active generative models at `7.62B + 0.51B = 8.12B <= 33B`:
   ```bash
   DIRECTOR_PRESET=qwen bash brev/serve.sh
   QWEN_DIRECTOR_BASE_URL=http://127.0.0.1:8001/v1 uv run --extra gym python -m gym.run_ablation --all
   ```

### Ablation metrics (live runs pending Brev endpoints; local stub validation artifacts in `results/ablation_stub_*.json`)

| Ablation Config | Endpoint Env Var | Swapped Component | Director Model | Total Gen Params (card / tensors) | Prompt | Tools | Thinking | Tool-Call Validity | Task Success | Steps to Report | Time to Report (ms) |
|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|
| `prompt_v1` | `DIRECTOR_BASE_URL` | System prompt (`v1`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 30.51B / 32.09B | `system_v1` | `verbose` | `off` | TBD (live run pending) | TBD (live run pending) | TBD (live run pending) | TBD (live run pending) |
| `terse_tools` | `DIRECTOR_BASE_URL` | Tool schemas (`terse`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 30.51B / 32.09B | `system_v2` | `terse` | `off` | TBD (live run pending) | TBD (live run pending) | TBD (live run pending) | TBD (live run pending) |
| `thinking_on` | `DIRECTOR_BASE_URL` | Thinking (`on`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 30.51B / 32.09B | `system_v2` | `verbose` | `on` | TBD (live run pending) | TBD (live run pending) | TBD (live run pending) | TBD (live run pending) |
| `nemotron_lightning` | `DIRECTOR_BASE_URL` | Reference (`v2` + `verbose` + `off`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 30.51B / 32.09B | `system_v2` | `verbose` | `off` | TBD (live run pending) | TBD (live run pending) | TBD (live run pending) | TBD (live run pending) |
| `qwen_director` | `QWEN_DIRECTOR_BASE_URL` | Model (`Qwen2.5-7B`) | `Qwen2.5-7B-Instruct` | 8.12B / 8.12B | `system_v2` | `verbose` | `off` | TBD (live run pending) | TBD (live run pending) | TBD (live run pending) | TBD (live run pending) |

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
   The alert card leads with the verified claim; the Analyst's words appear
   under it only when they name the same object.
3. **Recovery.** Press **Break the next tool call** (or corrupt the next
   tool's arguments with `curl -X POST 127.0.0.1:8000/demo/corrupt/1`):
   the Director gets an error, recovers, still reports.
4. **Fallback.** Cover the hero lens: the investigation finishes on scout1.
5. **Close** on the ratio: frames handled by a 10 ms model vs. VLM calls made.

### Test fault: a Scout false positive (`SCOUT_MISLABEL`)

`SCOUT_MISLABEL=cup=knife` renames a raw detector class before the Scout
classifies it, so a cup reaches the Director as a knife, exactly as a real
detector mistake would (nearest person and head attached). It is off unless
set; separate several renames with commas (`cup=knife,bottle=scissors`). It
works with every Scout backend, including the scripted scene. Use it for tests
and demos only.

Sherlock announces it so nobody watching takes it for a real detection: a
`system` row `Scout mislabel injected: cup -> knife (test fault, SCOUT_MISLABEL)`
at startup (console log; the board shows only events after it connects), and
`Scout mislabel injected: cup -> knife on hero (test fault, not a real
detection)` before each injected detection, in the console and the board's
event feed.

With real models, the scripted room shows the Verifier catching the Scout: the
cup's "a knife" claim is rejected, the report becomes a log entry with no face
crop kept, and seconds later the scene's real knife is confirmed.

```bash
SCOUT_MISLABEL=cup=knife uv run python run.py --synthetic
```

## Open data and synthetic dataset provenance

Sherlock incorporates both existing open datasets and openly published synthetic datasets built during the hackathon:

1. **Existing open data (COCO 2017 CC-BY subset, `scripts/build_scout_dataset.py`)**:
   - Filters `annotations/instances_train2017.json` and `instances_val2017.json` (CC-BY-4.0 annotations) to images carrying Flickr licenses `4` (Attribution), `7` (No known copyright restrictions), or `8` (US Government Work) via `scripts/scout_data.py` (`CC_BY_LICENSES`).
   - Maps COCO `person`, `knife`, `scissors`, `baseball bat`, and `bottle` annotations onto the four Scout runtime classes (`person`, `weapon`, `badge`, `bottle`) and records per-image `coco_id`, `license`, `license_name`, `coco_url`, and `flickr_url` in `data/coco_subset/manifest.jsonl`.
2. **Synthetic evaluation dataset (`gym/resources_servers/sherlock/data/tasks.jsonl` and `gym/resources_servers/sherlock/scenes.py`)**:
   - Contains 8 deterministic multi-step camera investigation scenarios generated from the state-machine simulator in `gym/resources_servers/sherlock/scenes.py` and released under the repository MIT [`LICENSE`](LICENSE):
     - Benign objects (`cup`, `bottle`, `cell phone`) requiring single-step `report(ignore)` with zero VLM calls.
     - Weapon escalations (`knife`, `scissors`, `baseball bat`) requiring weapon zoom + weapon crop + head zoom + unanalyzed face crop + `verify` confirmation + `report(alert)`.
     - Fault-injection and false-positive recovery scenarios (`look_at` hardware error, occluded `hero` camera requiring fallback to `scout1`, and unconfirmed Verifier claims requiring downgrade to `log`).
3. **VLM bake-off evaluation crops (`tests/fixtures/vllm/`)**:
   - Uses public-domain NASA imagery (`shuttle_crop.jpg`, documented in [`tests/fixtures/vllm/README.md`](tests/fixtures/vllm/README.md)) and 50 CC-BY COCO 2017 object crops (`knife`, `scissors`, `baseball bat`, `cup`, `bottle`) to fit the Verifier's yes/no log-odds thresholds (see "Verifier verdicts").

## Known limitations & next steps

### Known limitations
- **Human-in-the-loop review required for security alerts**: Small vision-language models (`SmolVLM-Instruct` 2.25B and `SmolVLM-500M-Instruct` 0.51B) are decision-support tools, not autonomous safety-critical detectors. In our 50-crop COCO bake-off, the yes/no `Verifier.check` at its default threshold confirms 27/30 real weapon crops on 500M and 29/30 on 2.2B, and accepts 1/20 and 0/20 false `"kitchen knife"` claims on cups and bottles. Borderline crops near the log-odds threshold can still flip across JPEG re-encodes, so every `alert` attaches both the weapon crop and an unanalyzed human-only face crop for operator verification.
- **The 500M Analyst names the biggest thing in a cluttered crop**: asked "What is this object?", SmolVLM-500M named the wrong object on 34 of 70 COCO weapon crops (a knife on a dinner table comes back "Pizza." or "Cake."), and three reworded prompts did no better (34 to 41 of 70). Alerts do not depend on it, because the Verifier is asked about the weapon itself. When a verified claim and the Analyst's words name different objects (21 of the 57 weapon crops the Verifier confirmed), the report keeps the words under `analyst_mismatch` and the dashboard shows the claim with "named another object · not shown" in their place.
- **Parameter-count accounting across model card vs. checkpoint tensors**: The default configuration, `NVIDIA-Nemotron-3.5-Lightning` (30B) + the shared `HuggingFaceTB/SmolVLM-500M-Instruct` (`0.51B`), totals **30.51B by model card and 32.09B by base checkpoint tensors** (raw BF16 safetensors headers excluding never-loaded `mtp.*` weights: `31.58B + 0.51B`), under 33B on both counts. The previous `SmolVLM-Instruct` (2.25B) default totals 32.25B by model card but `31.58B + 2.25B = 33.82B` by tensors, over the cap; the 500M costs single-crop Verifier recall (27/30 real weapon crops confirmed vs. 29/30 at the default Verifier threshold).
- **Platform-specific USB gimbal control**: Direct UVC pan/tilt/zoom control for the Insta360 Link 2 (`2e1a:4c01`, `2e1a:4c02`, `2e1a:4c04`) relies on macOS IOKit via `link-ctl`. On Linux, the stock `uvcvideo` kernel driver rejects UVC pan/tilt `SET_CUR` transfers without detaching the video stream, so Linux hosts automatically fall back to the secondary camera or software crop (`FakePTZ`).
- **Local dashboard network scope**: The FastAPI dashboard serves live MJPEG streams and saved evidence crops without authentication and therefore binds to `127.0.0.1` by default (`DASHBOARD_HOST=127.0.0.1`).
- **NeMo Gym simulation scope**: The 8 tasks in `gym/resources_servers/sherlock/data/tasks.jsonl` evaluate the Director's multi-turn tool policy, fault recovery, and verification discipline against deterministic scene responses rather than live pixel crops.

### Next steps
- **Revisit `InternVL3-1B` (0.94B) only if room photos disappoint**: it fits the 1.42B tensor-count VLM ceiling (`31.58B + 0.94B = 32.52B <= 33B`) and scored 95/100 on the bake-off, but its default 13-tile preprocessing costs about 3.6x SmolVLM-500M's input tokens, and the 500M matches it with the yes/no Verifier.
- **Expand in-room prop and lighting coverage**: Capture multi-angle in-room sessions across varied lighting with `scripts/capture_props.py` and `scripts/review_labels.py`, then publish the CC-BY dataset and fine-tuned Scout weights to Hugging Face via `scripts/publish_scout_hf.py --public`.
- **Continuous multi-camera handoff**: Extend `Orchestrator` from reactive single-incident failover (`hero` -> `scout1`) to continuous cross-camera homography calibration so `scout1` bounding boxes map directly into `hero` pan/tilt angles without an intermediate wide sweep.

## License

Code and synthetic evaluation datasets (`gym/resources_servers/sherlock/data/tasks.jsonl`) are released under the [MIT License](LICENSE). Vendored dashboard fonts carry the [SIL Open Font License 1.1](sherlock/sherlock/dashboard/static/fonts/OFL-ibm-plex-mono.txt). COCO 2017 annotations are CC-BY-4.0 with per-image Flickr attribution recorded in `data/coco_subset/manifest.jsonl`.

