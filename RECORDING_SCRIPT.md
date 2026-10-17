# Recording script: YOLO → Brev → camera zoom

About 4:00. Record a successful run ahead of time and replay it in the browser
for narration. Lead with the camera response. Keep the first detection-to-zoom
sequence at its original speed, in one continuous shot. Explain the model after the
viewer has seen the camera react.

**Length and additions.** The submission requires 3:00 to 5:00, on unlisted
YouTube or Vimeo, with the presenter camera bubble on to show the team. The
original take ran about 2:30. Sections marked **(added)** bring it to about
4:00: the team introduction, the Verifier catching an injected Scout mistake,
and the measured numbers. The original sections keep their wording; only their
times moved.

## Playback plan

Capture the board with its actual camera feeds and event timing. Play that video
in the browser for the opening and model explanation. Keep the original elapsed
time through the first close-up; trimming idle time before detection is fine.
Do not speed up the round trip to make it look faster.

Then cut to the working phone and custody tabs for the same captured incident.
Keep its evidence files and signed records available. A video itself is not
clickable: interact with the actual case pages for those shots. Browser event
replay would be a separate feature; it is not required for this recording plan.
Keep playback instructions out of the spoken narration. Use present tense
without claiming “live” or “happening right now.” Narrate the execution captured.

**(added)** Capture a second run for the Scout-mistake section, on the scripted
scene with a test fault that makes the Scout call the cup a knife. With the
Brev endpoints in `.env`:

```bash
SCOUT_MISLABEL=cup=knife VERIFIER_MARGIN=-1.0 \
  CAMERAS="hero:fake:synthetic,scout1:fake:synthetic" \
  uv run python run.py --synthetic
```

Capture from the second scene loop: the first request warms the models.
Keep the `Scout mislabel injected` row in frame when it appears.

## Before recording

Use the main flow board at `http://localhost:8000/`, framed at 1920×1080.
Keep `/phone` and `/custody` ready in separate tabs. Use the camera-control arrow
and the close-up feed as the visual focus for the opening.

Confirm the actual Scout is YOLO, the Director endpoint on Brev serves
`RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8`, and the camera responds to
its tool call before recording this narration. The repository now selects that
checkpoint; loading and testing it on the H100 are still required. A configured
model label alone does not establish that it ran.

Measure the opening from YOLO's detection event to the first returned close-up,
including the Brev round trip and camera action. Report that complete interval
only after measuring it. Model inference time alone is not camera response time.
For physical PTZ, show physical movement; for software cropping, say “close-up.”
Do not speed up this sequence or splice detection and response from separate runs.

## 0:00–0:25 · Show the camera react

**Screen:** Main board. Introduce the object, then let the detection, Brev
Director and return camera-control arrow play through to the close-up.

> Watch the camera. YOLO spots an object locally, sends the detection to the
> Director on Brev, and the Director tells the camera where to look.

**Pause as the close-up arrives. Leave it visible for two seconds.**

> There. We went from the room to the detail we need, in one loop.

If measured, add: “That took [measured interval] from detection to close-up.”
Otherwise let the uninterrupted action demonstrate the response time.

## 0:25–0:40 · Introduce the team (added)

**Screen:** Main board, presenter camera bubble in a corner. The bubble stays
on for the whole video.

> We're [team names], and this is Sherlock: a small detector watches every
> frame, and bigger models only run when something is worth a closer look.

Use the names of everyone on the team. Keep it to one sentence.

## 0:40–1:00 · Explain the gate

**Screen:** Keep the camera, Scout and Director visible. Show the benign object
if it appears naturally; do not delay the opening payoff to wait for it.

> We're monitoring a building with multiple cameras. We don't need to send
> every frame to a large model. YOLO does the initial pass locally. The Director
> decides whether to ask the camera for a closer look. A benign cup can stop
> here, with zero vision-model calls.

## 1:00–1:20 · Explain the FP8 choice

**Screen:** Director box and NVIDIA Brev heading.

> The Director is Nemotron 3.5 Lightning, using Red Hat AI's FP8 checkpoint
> from Hugging Face, served with vLLM on Brev. Red Hat used LLM Compressor to
> quantize it. FP8 uses eight bits for the quantized weights instead of sixteen
> in the BF16 version. It's the same model architecture and parameter count,
> with less storage per quantized weight.

This is the text-and-tool Director. SmolVLM handles the images. Do not call the
FP8 checkpoint a vision model, say it has half the weights, call it lossless,
or claim a measured speedup without a comparison. The previous repo default was
already NVFP4; the sixteen-to-eight comparison is with the BF16 base model.

## 1:20–1:45 · Verify the object

**Screen:** Close-up, then the Verifier and Analyst results.

> The Verifier checks the object claim using the masked close-up. Then the
> Analyst describes the object. The Verifier decides whether the claim is
> supported; the description gives the responder more context.

> The face crop and original frame stay with the evidence for a human
> reviewer. The models only receive the masked object crop. The reviewer can
> see who was holding the object and where it was in the scene.

**(added)** If a face falls inside the close-up, the crop shows a flat grey
patch there and the snapshot row reads `1 face masked`. Point at it: the face
is filled before any model sees the crop.

## 1:45–2:15 · The Verifier catches the Scout's mistake (added)

**Screen:** The second captured run. The `Scout mislabel injected` row, the
investigation, `REJECTED` from the Verifier, `face crop discarded: weapon
claim not verified`, the incident logged rather than alerted; then the real
knife's alert.

> Small detectors make mistakes, so here we inject one: the Scout is told to
> call this cup a knife. The Director investigates it like any knife. The
> Verifier looks at the masked crop and says no, so Sherlock logs it for a
> person instead of raising an alert, and the face crop is discarded. A few
> seconds later the real knife gets a yes.

Always say the mistake is injected; it is a test fault, not a detection.

## 2:15–2:45 · Show the human response

**Screen:** Click the phone indicator, select the notice, page through the
images, then choose Investigate and Notify team.

> The responder gets a notice with the building location and the pictures.
> They can inspect the scene, choose to investigate, and notify their team.
> This phone view demonstrates that response workflow.

The phone delivery is a demonstration, not a real external message. Keep the
app's response status visible; do not imply a real team received a notification.

## 2:45–3:30 · Show what we measured (added)

**Screen:** The MEASURED card on `/log`, then the "Ablation metrics" table in
the README.

> We benchmarked the Director on vLLM on the H100, using the NVFP4 checkpoint
> of the same model. One request at a time, it returns its tool call in about
> 230 milliseconds at the median, and 99 percent of its tool responses are
> valid. At sixteen at once it serves twelve requests a second.

> Then we used NeMo Gym to test the Director on eight scripted incidents with
> faults. The prompt is what matters: the old prompt fails all eight, ours
> passes all eight in about a second and a half each. Thinking on passes the
> same eight but takes more than twice as long, so it ships off. A Qwen 7B
> Director also passes all eight, so this suite does not rank them.

These H100 numbers ran the NVFP4 checkpoint, before the switch to FP8
(README, "H100 results" and "Ablation metrics"). If FP8 measurements land
before recording, use them and drop the NVFP4 clause. Never claim a
prefix-cache gain (the A/B showed none) or that Nemotron beats Qwen. The card's Director headline is at sixteen
concurrent requests; do not read it as the one-at-a-time figure.

## 3:30–4:00 · Show the evidence and close

**Screen:** Return to the board and click an evidence image for the summary.
Open its full case file, expand the incident and show its timeline and signature.

> We preserve the captured evidence whether or not the incident becomes an
> alert. Each crop links back to its source frame. The case file records the
> camera, the analysis, the decision and the responder's actions, with image
> hashes and signed revisions that we can verify.

> Expanso carries the pipeline events and evidence between the stages. The
> goal is to spend bandwidth and model work on the moments that need a closer
> look, and give a person the context they need to act.

Say the Expanso sentence only with a verified connected run and downstream
receipt. Otherwise use: “Expanso provides the transport for this architecture.”
The signatures establish consistency with the local application key; do not
claim independent attestation. Do not use a blanket “under two seconds,” cost
saving or false-positive reduction figure without evidence from this run.

## Optional extra, after the main take

Show a failed camera tool call and recovery to the other camera. Keep this out of
the opening: the first payoff is the fast YOLO → Brev → camera response.

## Model source

[Red Hat AI FP8 model card](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8),
checked 2026-10-17: text input/output, FP8 weights and activations, BF16 base
checkpoint, LLM Compressor, and vLLM serving with the Nemotron tool parsers.
