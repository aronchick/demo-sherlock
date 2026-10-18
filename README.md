# Sherlock: a camera that investigates

Sherlock detects an object, asks a model whether to take a closer look, moves
the camera, and checks the resulting evidence. A reviewer gets the scene,
close-up, camera location and signed incident history.

Built for the NVIDIA × Red Hat Small Models Hack, Raleigh, October 2026.
The demo combines a local YOLO Scout, a Nemotron Director, a shared SmolVLM
Analyst/Verifier, and Expanso Cloud-managed data pipelines.

## How it works

The current camera setup uses one **OBSBOT Tiny 2 Lite** connected to a Mac.
The Mac runs detection, camera control, evidence storage and the presenter.
A Brev GPU serves the generative models. A second camera is optional; the
application does not assume a MacBook webcam or an Insta360 hero camera.

```text
OBSBOT Tiny 2 Lite
    │ video                         ▲ pan / tilt / zoom
    ▼                               │
Mac: fine-tuned YOLO Scout → Orchestrator + camera tools
    │                             │              ▲
    │ ordinary object             │ ticket       │ tool calls
    ▼                             ▼              │
Local ignore record          Brev: Nemotron Director
                                  │
Mac: capture close-up ◄────────────┘
    │ masked object crop
    ▼
Brev: SmolVLM Verifier → SmolVLM Analyst
    │ verdict + description
    ▼
Mac: signed incident → board / phone / evidence register
    │ metadata + masked object crop
    ▼
Expanso Edge → Brev core storage → hash-checked receipt
       Jobs deployed and monitored through Expanso Cloud
```

1. **Detect locally.** The bundled fine-tuned YOLOv8n checkpoint detects
   room props, including caps and fedoras. Weapon and hat candidates use a
   dwell capped at 120 ms; ordinary objects use the configured dwell
   (600 ms by default). These are trigger delays, not end-to-end latency.
2. **Choose a response.** Ordinary objects are handled locally without a
   generative-model call. Weapons and hats go to the Director, which calls
   camera tools to acquire a close-up. Hats are inspections, never weapon
   alerts. People provide spatial context and do not trigger investigations.
3. **Check the crop.** After acquisition, the Verifier and then the Analyst
   inspect the same face-masked object crop. The Verifier checks the claim;
   the Analyst supplies a description. Only a confirmed weapon claim can
   produce an alert for human review. Rejected, inconclusive and failed
   investigations remain logged.
4. **Preserve the evidence.** The original scene and available object crop
   stay in the local incident record. After a confirmed weapon alert, the
   application can retain a face crop from the original scene for the human
   reviewer. The current Director prompt does not request a second camera
   move to the face. Neither the original scene nor the face crop goes to a
   model or the Expanso incident pipeline.
5. **Deliver and verify.** Expanso transports incident metadata and the
   masked object crop to the core. A separate reader checks stored crop bytes
   and returns a receipt; the dashboard matches it against local evidence.
   Pipeline acceptance alone does not establish remote storage.

Camera movement and inference are separate stages. The loop has no fixed
10 ms frame-time or two-second investigation guarantee. The board reports
measured timings for the current run.

## Models and configuration

| Role | Current default | Runs on |
| --- | --- | --- |
| Scout | Fine-tuned YOLOv8n, `models/scout-r2-960.pt` | Mac |
| Director | `RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8` | Brev, vLLM `:8001` |
| Analyst and Verifier | One shared `HuggingFaceTB/SmolVLM-500M-Instruct` | Brev, vLLM `:8002` |

[`brev/serve.sh`](brev/serve.sh) pins vLLM 0.28.0 and starts two containers.
The Director uses tool calling, thinking off and prefix caching on by default.
No speculative draft model is loaded. Generative parameter counts are 30.51B
by model card and 32.09B by base checkpoint tensors, excluding unloaded MTP
weights. YOLO is a separate, non-generative detector.

The Scout runs at 640 pixels with weapon confidence 0.15, other-class
confidence 0.35 and NMS IoU 0.6. Its 960-pixel training resolution is not its
runtime input size. See [checkpoint provenance and license](models/README.md)
and [evaluation limits](docs/SCOUT_R2_VALIDATION.md).

The Verifier uses the yes/no log-probability margin when available, otherwise
it reads the reply's first word. `VERIFIER_MARGIN` defaults to 0.0 in code;
the supplied demo configuration uses -1.0. An Analyst description cannot
overrule the Verifier. Conflicting descriptions remain in the record.

## Run the demo

Install [uv](https://docs.astral.sh/uv/) and
[just](https://github.com/casey/just). Run commands from this checkout.
`uv` installs the locked Python dependencies on first use.

### Rehearse without cameras or a GPU

```bash
just demo-up
```

Open <http://127.0.0.1:8000>. This uses generated video, a rule-based Director
and mock vision responses. It exercises the application, not real detection
or model quality. Stop the presenter with:

```bash
just down-local
```

### Use the real camera and Brev models

Start from [`.env.example`](.env.example) and edit your project `.env`.
For the demonstrated OBSBOT, use:

```dotenv
CAMERAS=hero:obsbot:name:OBSBOT Tiny 2 Lite StreamCamera
YOLO_MODEL=models/scout-r2-960.pt
SCOUT_WEAPON_CONF=0.15
SCOUT_IMGSZ=640
SCOUT_NMS_IOU=0.6
CAMERA_LOCATIONS='{"hero":{"area":"Your actual room"}}'
```

The example file uses software-crop cameras so it does not assume that
specific hardware is present. The `fake` camera backend means software PTZ
on a real video source; `--synthetic` selects generated video.

Full startup requires an existing Brev GPU instance, authenticated Brev CLI,
Expanso CLI/Edge binaries and workspace credentials. Put
`EXPANSO_CLI_ENDPOINT`, `EXPANSO_CLI_API_KEY` and
`EXPANSO_EDGE_BOOTSTRAP_TOKEN` in an owner-only project env file (mode 0600).
`EXPANSO_WORKSPACE_ENV` selects that file explicitly. Do not commit credentials.

Select your existing instance in the same shell used for startup and shutdown:

```bash
export BREV_INSTANCE=your-existing-instance
just up
```

`just up` starts the existing instance, configures model serving and the
Expanso core, verifies inference through an SSH tunnel, starts the laptop
Edge node and launches the localhost presenter. It reuses matching healthy
services and does not provision a new GPU instance. Without `BREV_INSTANCE`,
the project default is `sherlock-nebius-h100`.

```bash
just down
```

`just down` stops owned local services and requests GPU shutdown, then checks
Brev's stopped state. Storage charges can remain. `just down-local` stops
only the presenter; the GPU, tunnel and Edge services remain running. Use
`just down-local` followed by `just up` after configuration changes.

For existing model endpoints without managed Cloud startup, configure the
`DIRECTOR_*`, `ANALYST_*` and `VERIFIER_*` values and run `just live`.
See [Brev setup and benchmarking](brev/README.md),
[Brev CLI workflow](brev/CLI.md) and
[OBSBOT hardware measurements](docs/OBSBOT_HARDWARE.md).

## Review an investigation

All presenter and recording pages stay on localhost. The application serves
camera feeds and evidence without authentication; do not expose it publicly.

| Page | Purpose |
| --- | --- |
| `/` | Live camera-to-decision board |
| `/phone` | Simulated responder device: investigate, notify, resolve |
| `/custody` | Evidence register, provenance, signatures and revisions |
| `/topology` | Configured camera/model placement and serving metrics |
| `/slides` | Six-slide demo briefing |
| `/log` | Event-oriented board |

The phone's actions persist locally as signed revisions. They do not send
SMS, push notifications or team messages. An optional core alert webhook is
a separate integration. Camera locations are operator-supplied labels, not
positions inferred by a model.

Evidence lives under `snapshots/` (`EVIDENCE_DIR` overrides it). Each signed
revision links image hashes, capture provenance, model results and responder
actions with Ed25519 signatures. Content-addressed images preserve earlier
revisions. Protect the local `.signing-key` alongside the evidence store.
These signatures establish local record integrity, not independent camera
identity, trusted timestamps or immutable storage.

## Expanso data flow

Model requests use direct OpenAI-compatible HTTP over the SSH tunnel.
Expanso carries the resulting events and evidence through five jobs:

| Job | Location | Data |
| --- | --- | --- |
| `sherlock-telemetry` | Mac | Text events with image payloads removed |
| `sherlock-incidents` | Mac | Incident metadata and masked object crops |
| `sherlock-examples` | Mac | Unconfirmed object crops; opt-in only |
| `sherlock-core-intake` | Brev | Stored records and optional alert webhook |
| `sherlock-vllm-metrics` | Brev | Serving metrics returned to the dashboard |

[Job definitions](jobs/) are deployed by Expanso Cloud to labeled Edge
nodes. The sink is enabled by `SHERLOCK_EDGE=1`, which `just up` sets.
Training-example export separately requires `SHERLOCK_EDGE_EXAMPLES=1`.
Failed core delivery can spool locally; it is not a storage receipt.
[`core_receipts.py`](scripts/core_receipts.py) reads stored crop bytes and
returns their hashes through the loopback tunnel. Receipt matching is a
trusted-peer check, not an independently attested storage service.

## Evidence and limitations

The repository retains historical measurements in [results/](results/README.md).
They describe their recorded models and inputs, not the health or speed of a
current deployment.

- **Scout:** the selected checkpoint detected 35 of 40 labeled weapons in
  the 122-image room test subset, with four false-positive boxes. Nearby
  video frames span splits, and thresholds were informed by those images;
  this is not an independent deployment benchmark. Broader-image regressions
  and misses are documented in [Scout validation](docs/SCOUT_R2_VALIDATION.md).
- **Camera:** [OBSBOT tests](docs/OBSBOT_HARDWARE.md) measured physical
  control and settling separately from inference. A [recorded live run](docs/REALTIME_VALIDATION.md)
  completed in 2.22 seconds; its older detector settings are historical.
- **vLLM:** the retained [H100 prefix-cache experiment](results/brev_preflight_20261017T200806_125411Z/)
  used the earlier NVFP4 Director, not today's FP8 default. It found no
  measurable cache benefit. It was one off-then-on run, not a repeated trial.
- **NeMo Gym:** [eight scripted tasks](gym/resources_servers/sherlock/data/tasks.jsonl)
  test tool policy and recovery, not live vision. The [Brev results](results/brev_vllm/)
  used the earlier NVFP4 checkpoint for Lightning. The reference passed 8/8;
  the older prompt passed 0/8. Those results do not validate the FP8 default.
- **Human review:** the Verifier can miss weapons and accept incorrect
  object claims. The Analyst can describe the wrong object. A confirmed
  alert means a possible weapon for human review, not intent or danger.
- **Camera coverage:** fallback needs another configured, working camera.
  The single-OBSBOT setup has no second-camera failover. Moving props can
  leave the crop while the gimbal settles.

## Development

```bash
just check
```

This runs Ruff and the test suite. Optional Torch/Ultralytics tests require
`uv run --extra yolo pytest`. NeMo Gym uses its `gym` extra on Python 3.13.14
or later. There is no hosted CI workflow in this repository.

Use `just bench-stub` or `just preflight-stub` to exercise benchmark plumbing
without a GPU. Stub results are not model measurements. Real benchmark setup
is documented in [brev/README.md](brev/README.md).

Scout data preparation, label review, training and evaluation live in
[`scripts/`](scripts/): `build_scout_dataset.py`, `capture_props.py`,
`review_labels.py`, `train_scout.py` and `eval_scout.py`. Each exposes `--help`.
Only the Scout is fine-tuned; NeMo Gym evaluates Director behavior without
changing weights. Room photographs are not distributed with this repository.

## License

Code and synthetic Gym scenarios use the [MIT License](LICENSE). The bundled
Scout checkpoint derives from Ultralytics YOLOv8n and carries
[AGPL-3.0 terms](models/LICENSE-ultralytics); MIT does not relicense it.
The [YuNet detector](sherlock/sherlock/models/LICENSE-yunet) and vendored
[IBM Plex](sherlock/sherlock/dashboard/static/fonts/OFL-ibm-plex-mono.txt) and
[Schibsted Grotesk](sherlock/sherlock/dashboard/static/fonts/OFL-schibsted-grotesk.txt)
fonts retain their own licenses.

The COCO preparation script filters eligible image licenses and records
per-image attribution in its generated manifest. The bundled
[NASA test fixture](tests/fixtures/vllm/README.md) is public domain;
the historical COCO evaluation crops and private room dataset are not bundled.
