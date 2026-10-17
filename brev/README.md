# Brev H100 preflight and evidence run

Run this from the repository root on an existing Brev H100. The machine needs
Docker with GPU access, uv, and access to the model repositories. This workflow
does not create or provision a machine. The benchmark script and its JPEG fixture
from PR #11 must be present before a sweep or A/B can run.

## Selected checkpoint

`brev/serve.sh` now defaults to
`RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8` for the Director.
The [model card](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8)
documents FP8 weights/activations and the same `qwen3_coder` tool parser and
`nemotron_v3` reasoning parser. The shared image model remains SmolVLM-500M.
The H100 load and end-to-end camera loop must be checked for this checkpoint;
previous NVFP4 results do not establish its latency or quality. Existing running
containers are not changed by editing this repository.

## One command on the GPU

For the full prefix-cache comparison, replace `my-instance` with the Brev instance
name and run:

```bash
uv run scripts/brev_preflight.py --mode ab --instance my-instance
```

This command starts the servers itself. It also works after `bash brev/serve.sh`:
it stops and removes the existing `director` and `vlm` containers before the
baseline, checks their configured vLLM image first, and retains the Hugging Face
cache mount. Run it when those two containers are available for the experiment.
It leaves the cache-on candidate serving for the demo.

The sequence is explicit:

1. Stop/remove only the named `director` and `vlm` containers, if present.
2. Run `PREFIX_CACHING=0 bash brev/serve.sh`.
3. Poll both `/v1/models` endpoints, then print their served model IDs. Each
   endpoint has a 30-minute readiness deadline; authentication errors fail early.
4. Send a weapon-scene Director request with `enable_thinking: false`. Validate
   its complete tool calls against the application's imported schemas. Reject
   reasoning, unknown tools, invalid arguments, prose-only and truncated replies.
   Send the bundled real JPEG object crop to the shared VLM and require a complete,
   nonempty response. No camera tools execute during this smoke check.
5. Run both benchmark workloads at concurrency 1, 4, 8 and 16, with 200 measured
   requests and four warmups per cell; attempt `/metrics` collection.
6. Stop/remove the same two containers, then run
   `PREFIX_CACHING=1 bash brev/serve.sh`.
7. Repeat readiness, smoke and sweep, comparing the candidate JSON to the baseline.
   Check the container image IDs and serving arguments stayed identical apart
   from the Director prefix-cache flags. Reject speculative-decoding flags.

The app prompts and tool schemas are imported at runtime, including weapon-flow
changes. The smoke result checks the API contract; it does not establish weapon
recognition accuracy or prove that the whole investigation chain works.

To test already-running servers without restarting them:

```bash
uv run scripts/brev_preflight.py --mode sweep --prefix-caching on
```

Here `--prefix-caching on` records the operator's declaration; it does not change
the server. `--mode smoke` sends only the two smoke calls. `--image path/to/crop.jpg`
replaces the bundled crop. Set `DIRECTOR_BASE_URL` and `ANALYST_BASE_URL` to override
the default loopback ports; `ab` requires local ports 8001/8002 on the GPU host.

## Connect the Mac

The preflight prints exact `.env` lines using the model IDs it selected. By default
they use loopback for the SSH tunnel below:

```dotenv
DIRECTOR_BASE_URL=http://127.0.0.1:8001/v1
DIRECTOR_MODEL="director"
DIRECTOR_THINKING=0
ANALYST_BASE_URL=http://127.0.0.1:8002/v1
ANALYST_MODEL="vlm"
VERIFIER_BASE_URL=http://127.0.0.1:8002/v1
VERIFIER_MODEL="vlm"
VERIFIER_MARGIN=-1.0
SHERLOCK_MOCK=0
OPENAI_API_KEY=EMPTY
```

Use `--mac-host HOST` on the GPU command to print direct-host URLs instead. If
direct ports are unreachable at the venue, start these on the Mac in **two
separate terminals**, one command per terminal:

```bash
brev port-forward my-instance --port 8001:8001
```

```bash
brev port-forward my-instance --port 8002:8002
```

Or use one SSH session, after refreshing Brev's SSH configuration:

```bash
brev refresh
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8001:127.0.0.1:8001 \
  -L 127.0.0.1:8002:127.0.0.1:8002 my-instance
```

With the Expanso data plane (README: Expanso Edge data plane), the same
tunnel also forwards the core intake and lets the core's metrics pipeline
reach the dashboard. Both stay on loopback at each end:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8001:127.0.0.1:8001 \
  -L 127.0.0.1:8002:127.0.0.1:8002 \
  -L 127.0.0.1:23121:127.0.0.1:23121 \
  -R 127.0.0.1:23100:127.0.0.1:8000 my-instance
```

The `-R` mapping assumes the dashboard on port 8000; change the last `8000`
if `DASHBOARD_PORT` differs. The dashboard accepts serving records only from
loopback callers.

Keep the tunnel running and use the loopback `.env` values. Both local ports must
be free before forwarding. From the Mac repository root, verify reachability:

```bash
uv run scripts/brev_preflight.py --mode models --ready-timeout 30
```

This reads `.env`, checks both APIs and prints model IDs without generating tokens.
The existing application uses one `OPENAI_API_KEY` for all roles. The preflight
also accepts `DIRECTOR_API_KEY` and `ANALYST_API_KEY` for testing, but never prints
their values. Set real credentials privately if the servers require them.

Start `just live` on the Mac and open <http://127.0.0.1:8000> there (or the port
set by `DASHBOARD_PORT`). The dashboard binds to `127.0.0.1` by default so camera
feeds and face crops stay local during demos and recording. Override with
`DASHBOARD_HOST=0.0.0.0` in the shell or `.env` **only on a trusted network**;
this deliberately allows LAN access to the dashboard. The model API tunnels
above do not require changing the dashboard bind address.

Brev syntax was checked against NVIDIA's [connectivity guide][connectivity] and
[connect command guide][connect]. The port mapping is `local:remote`; repeating
`--port` in one invocation keeps only the last mapping. Brev web-console public
tunnels require browser authentication, so use SSH forwarding for API clients.
The Brev CLI was not installed or exercised locally.

## Results and failure recovery

Each invocation writes `results/brev_preflight_<timestamp>/preflight.json` with
stage status, selected IDs, smoke outcomes and benchmark JSON paths. A/B writes
the benchmark JSON and Markdown tables beneath `off/` and `on/`; the `on` table
includes the comparison. The manifest and measurements carry `evidence_kind`.
Only actual endpoint runs support serving-performance claims. Local tests use
`--evidence-kind stub` and synthetic responses.

The sweep measures TTFT and end-to-end p50/p95/p99, output tokens/s, requests/s,
Director tool-call validity and optional cache/KV metrics. Benchmark failures
return a nonzero exit code and preserve their result files. A timeout also leaves
a partial manifest. Inspect `docker logs director` and `docker logs vlm` on the
GPU to diagnose startup failures; the preflight does not dump those logs or keys.
Fix the cause, then rerun the same command. It safely replaces the experiment's
named containers and uses a fresh output folder. It does not roll back a failed
startup; a failure can leave one or both servers down.

Keep model revisions and host load fixed during the experiment. Image/argument
hashes cannot detect a changed model repository revision behind an unpinned ID.
Repeat measurements to check noise; a single off/on pass does not establish a
stable speedup. When finished with the GPU, stop its containers explicitly:

```bash
docker container stop --timeout 30 director vlm
```

Stop the Mac tunnels with Ctrl-C. This workflow leaves provisioning and machine
shutdown to the operator.

[connectivity]: https://docs.nvidia.com/brev/cli/connectivity
[connect]: https://docs.nvidia.com/brev/cli/brev-connect
