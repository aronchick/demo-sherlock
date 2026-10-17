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
Rule-based director, mock VLMs, a scripted room on a 26 s loop: someone holds up a cup (ignored, 0 VLM calls),
then someone holds up a knife (look_at → snapshot → look_at head → face snapshot → verify → alert with both crops).
Press **Break the next tool call**, **Corrupt next tool args** or **Cover hero lens** on the dashboard to see recovery.
Evidence lands in `snapshots/<incident>/` (`EVIDENCE_DIR` overrides). The face crop is stored for a human only:
no model ever sees it, and nothing identifies or describes the person.

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

## vLLM on Brev (bounty + speed)
Serve the Director (and optionally the VLM) with vLLM on a Brev GPU, then:
```
DIRECTOR_BASE_URL=https://<brev-host>/v1  DIRECTOR_MODEL=<nemotron>
```
Keep the Ollama values in `.env` as the one-line fallback if venue wifi dies mid-pitch.
If the endpoint is unreachable the Director automatically drops to the rule-based path.

## Layout
```
sherlock/camera/     Camera interface, FakePTZ (crop/zoom), ObsbotCamera (UVC), SyntheticSource
sherlock/agents/     scout.py (YOLO + motion fallback + dedupe/dwell), director.py (tool loop + RuleDirector), vision.py (Analyst, Verifier, mocks)
sherlock/tools.py    Tool schemas + dispatcher + failure injection
sherlock/orchestrator.py   Scout → investigate → verify loop, per-incident evidence, stats
sherlock/scene.py    Scripted synthetic room (cup, then knife) for --synthetic
sherlock/dashboard/  FastAPI SSE + MJPEG + the judge-facing UI
tests/               End-to-end mock run: alert chain, ignore path (zero VLM calls), evidence, recovery
```

## Development
The lint gate is one command, and it must report zero findings:
```bash
uvx ruff@0.17.1 check .
```
`ruff.toml` pins this exact version (`required-version`; other versions refuse to run) and spells out its rule set, so every machine gets the same answer.

## Demo script (3 min)
1. Dashboard up, wide view showing the room, counter climbing with zero VLM calls.
2. Judge holds up a cup: Scout tags it benign, Director ignores it, 0 VLM calls. Judge holds up a knife (or scissors):
   camera zooms on the weapon, Analyst describes it, camera zooms on the head for a face crop (never analyzed),
   Verifier checks the weapon crop, alert with both crops. Point at latency per stage.
3. Press **Break the next tool call** → Director gets an error, retries differently, still reports.
4. Cover the hero lens → Director lists cameras, finishes on scout1.
5. Close on the ratio: frames handled by a 10 ms model vs. VLM calls actually made.
