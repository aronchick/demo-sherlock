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
   OpenCV numbers cameras by sorted AVFoundation unique ID, so indexes reshuffle whenever a device (Link 2, iPhone Continuity Camera, OBS Virtual Camera) appears or disappears, and two cameras can silently swap mid-demo. `name:` matches the names `ptz_probe.py` prints (exact, case-insensitive, or a unique substring); a missing or ambiguous name fails loudly at startup. Numeric indexes still work.

Control path on macOS: standard UVC pan/tilt/zoom control transfers sent straight over USB with [link-ctl](https://pypi.org/project/link-ctl/)'s IOKit module (no sudo, no app). On Linux: `v4l2-ctl`. The Link Controller WebSocket API is not used because it cannot set an absolute pan/tilt. If the gimbal does not answer within `INSTA360_CMD_TIMEOUT_S`, `look_at`/`move` return a tool error and the Director moves to the next camera.

Tuning (env, defaults in `sherlock/camera/insta360.py`):

| Variable | Default | What it does |
| :--- | :--- | :--- |
| `INSTA360_FILL` | `0.6` | Share of the frame the target fills after `look_at` |
| `INSTA360_MAX_ZOOM` | `4.0` | Zoom cap (Link 2 zoom is 4x digital; lower it if crops get soft) |
| `INSTA360_HFOV_DEG` | `67` | Horizontal FOV at 1x, from the Insta360 spec sheet |
| `INSTA360_SETTLE_S` / `INSTA360_SLEW_DEG_S` / `INSTA360_ZOOM_SETTLE_S` | `0.35` / `120` / `0.1` | Wait before a snapshot: fixed + travel/speed + per-1x zoom change |
| `INSTA360_HOME_AFTER_S` | `6` | Return to the wide home pose after this many idle seconds (`0` = never) |
| `INSTA360_PAN_SIGN` / `INSTA360_TILT_SIGN` / `INSTA360_UNITS_PER_DEG` | `1` / `1` / `3600` | Direction and units of the raw UVC values (arc-seconds per spec) |

The OBSBOT adapter (`hero:obsbot:<n>`) is kept as an alternate backend.

## vLLM on Brev (bounty + speed)
Serve the Director (and optionally the VLM) with vLLM on a Brev GPU, then:
```
DIRECTOR_BASE_URL=https://<brev-host>/v1  DIRECTOR_MODEL=<nemotron>
```
Keep the Ollama values in `.env` as the one-line fallback if venue wifi dies mid-pitch.
If the endpoint is unreachable the Director automatically drops to the rule-based path.

## Layout
```
sherlock/camera/     Camera interface, FakePTZ (crop/zoom), Insta360Camera (Link 2 gimbal), ObsbotCamera (UVC), SyntheticSource
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
