# Sherlock — a camera that investigates

Small models that notice, decide, zoom, read, and verify. NVIDIA × Red Hat Small Models Hack, Raleigh, Oct 2026.
Tracks: Multi-Agent (primary), Tool-Calling, Nemotron + vLLM bounties.

```
Scout (YOLO nano, every frame, ~10 ms)
  └─ new object? ──► Director (small LLM + camera tools)
                        ├─ look_at / move  → PTZ camera (OBSBOT or software crop)
                        ├─ snapshot        → Analyst (2–3B VLM on the zoomed crop only)
                        └─ report          → Verifier (second VLM pass: does the crop support the claim?)
```
One tiny model gates the expensive ones. The dashboard counts "VLM calls avoided" live.

## Run with nothing (sandbox, airplane, dead wifi)
```bash
pip install -r requirements.txt            # ultralytics optional for this mode
CAMERAS=hero:fake:synthetic,scout1:fake:synthetic python run.py --mock --synthetic
open http://localhost:8000
```
Rule-based director, mock VLMs, generated video. Press **Break the next tool call** and **Cover hero lens** on the dashboard to see recovery.

## Run on a Mac with real models
```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct   # Director — swap for Nemotron 3.5 Lightning once a build fits
ollama pull qwen2.5vl:3b          # Analyst + Verifier (moondream / smolvlm also fine)
cp .env.example .env              # CAMERAS=hero:fake:0 uses the laptop webcam with software PTZ
python run.py
```

## When the OBSBOT arrives
1. `python scripts/ptz_probe.py` — checks whether the camera answers standard UVC pan/tilt/zoom.
2. If yes: `CAMERAS=hero:obsbot:0,scout1:fake:1`. Nothing else changes.
3. If no: Plan B is OSC via OBSBOT Center; implement `ObsbotCamera.move` against it.

## vLLM

vLLM is the serving layer for Sherlock's investigation loop. The Director
chooses camera tools; the Analyst reads a zoomed crop; the Verifier checks
the resulting claim against that crop. Scout gates these calls locally.
The planned Brev setup runs two vLLM **0.28.0** containers on one H100:

| Role | Model | Served name | Port |
|---|---|---|---:|
| Director | Nemotron 3.5 Lightning 30B, NVFP4 repository | `director` | 8001 |
| Analyst + Verifier | Small VLM, approximately 2B; exact revision pending | `vlm` | 8002 |

The Director uses automatic tool selection and disables thinking in the
request's chat-template kwargs. Analyst and Verifier share one VLM to avoid
loading a third model. Separate context, sequence and memory budgets let
both servers share the GPU. The initial `brev/serve.sh` budgets are 32,768
tokens / 16 sequences / 0.60 GPU memory for Director and 8,192 / 8 / 0.30
for VLM. These are starting settings, not measured capacity; the serving
owner must confirm model revisions, parser, startup flags and available
memory on the H100. This benchmark does not change `serve.sh`.

**Live H100 measurements are pending.** The benchmark is validated against
a local streaming stub. Stub results test the measurement code and must
not be submitted as vLLM performance evidence.

### Run the benchmark

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
`sherlock/agents/director.py` and `TOOL_SCHEMAS` from `sherlock/tools.py`.
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
Comparison rejects different workloads, settings, evidence types or models.
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
uv run --with-requirements requirements-bench.txt ruff check .
PYTHONPATH=sherlock uv run \
  --with-requirements requirements-bench.txt pytest -q
```

The tests start a tiny loopback OpenAI-compatible SSE server, run the
benchmark CLI through both A/B legs and all concurrency levels, verify JSON
and Markdown, stop the server and rebind its port to prove cleanup. They
also check HTTP errors, timeouts, truncated streams, missing usage,
invalid tool calls and counter resets. No Brev endpoint is needed.

## Layout
```
sherlock/camera/     Camera interface, FakePTZ (crop/zoom), ObsbotCamera (UVC), SyntheticSource
sherlock/agents/     scout.py (YOLO + motion fallback + dedupe/dwell), director.py (tool loop + RuleDirector), vision.py (Analyst, Verifier, mocks)
sherlock/tools.py    Tool schemas + dispatcher + failure injection
sherlock/orchestrator.py   Scout → investigate → verify loop, stats
sherlock/dashboard/  FastAPI SSE + MJPEG + the judge-facing UI
tests/               End-to-end mock run incl. tool failure recovery and covered-lens fallback
```

## Development
The lint gate is one command, and it must report zero findings:
```bash
uvx ruff@0.17.1 check .
```
`ruff.toml` pins this exact version (`required-version`; other versions refuse to run) and spells out its rule set, so every machine gets the same answer.

## Demo script (3 min)
1. Dashboard up, wide view showing the room, counter climbing with zero VLM calls.
2. Judge holds up an object. Scout flags it, camera whirs over and zooms, Analyst reads the label, Verifier confirms. Point at latency per stage.
3. Press **Break the next tool call** → Director gets an error, retries differently, still reports.
4. Cover the hero lens → Director lists cameras, finishes on scout1.
5. Close on the ratio: frames handled by a 10 ms model vs. VLM calls actually made.
