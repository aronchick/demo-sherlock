# Sherlock: three-minute recording script

Target 3:00, including pauses and screen changes. The spoken copy is about
350 words. Rehearse at 125–135 words per minute and let the camera movement
breathe. Use the five-slide briefing as bookends; spend most of the take in the
working product. Keep the presenter camera bubble clear of the event feed.

## Screen order and spoken copy

### 0:00–0:18 · Problem · `/slides#problem`

> We built Sherlock to help security teams investigate camera detections. A building can have dozens of cameras,
> but someone still has to decide which moments deserve attention. We want
> to get a closer look quickly, check the claim, and preserve the evidence
> for the person who responds.

### 0:18–0:28 · Set up the action · `/slides#camera-loop`

> Watch this loop: YOLO detects locally, Nemotron chooses a camera action
> on Brev, and the camera turns and zooms to capture the detail.

### 0:28–1:00 · Camera response · `/`

Introduce the prop into the wide view. Keep detection, the Director response,
and physical camera movement together in one uninterrupted shot at original
speed. Pause while it moves; hold the returned close-up for two seconds.

> YOLO sends a small detection ticket. The Director returns a tool call:
> where to look, and what to capture. There is the close-up, linked to the
> original scene. Ordinary objects can be logged locally, so they don't
> need the same cloud investigation.

### 1:00–1:25 · Explain the serving choice · board, then `/slides#serving`

> The Director is Nemotron 3.5 Lightning, served with vLLM on NVIDIA Brev.
> We're using Red Hat AI's FP8 checkpoint from Hugging Face, quantized with
> LLM Compressor. That reduces storage per quantized weight. The Director
> handles text and camera tools; SmolVLM handles the pictures. Each model
> has a specific job in this loop.

### 1:25–1:47 · Check the result · `/`

> The Verifier checks the object claim against the close-up. A supported
> claim can become an alert; an unsupported claim stays in the incident
> log. The Analyst adds a description. The source frame and object evidence
> stay attached, so the reviewer can see what led to the decision.

Show the real decision and description. Do not wait for a second incident.

### 1:47–2:12 · Human response · `/phone`

Select this incident, page through its images, then choose Investigate and
Notify team. Keep the building location and response status in frame.

> The phone view puts the location and pictures together. I can open the
> notice, page through the scene and the close-up, and mark it for
> investigation. Notify team records the next response action. The person
> gets the evidence they need to decide what to do.

### 2:12–2:42 · Evidence and transport · board modal, then `/custody`

Click this incident's image on the board for its summary. Open the full case,
expand its history, and show the image hash and signature. Show the transport
receipt for the same incident when verified.

> The case file keeps the capture, model decision and response history
> together. Image hashes and signed revisions let us check whether those
> records changed. Expanso carries the incident through the data pipeline;
> the downstream receipt shows that it arrived. We can trace this result
> back to the camera image that started it.

### 2:42–3:00 · Close · `/slides#impact`

> The aim is fewer frames sent for expensive analysis, fewer unsupported
> alerts, and enough context for a person to act. Those are benefits to
> measure against a full-frame baseline. The repository has the code,
> architecture and evaluation details so you can inspect how it works.

Hold the repository link until 3:00. The evidence slide at `/slides#evidence`
is a useful explanation or backup if a judge asks about custody.

## Recording preflight

- Use 1920×1080. Open the board, briefing, phone and custody in separate tabs.
  Pick one incident and follow its ID through every view.
- Confirm the current camera is OBSBOT, the Scout is YOLO, and the Brev model
  endpoints answer real inference requests. A model label is not execution proof.
- Verify the Expanso pipeline and downstream receipt for this run before using
  the transport sentence. If receipt is missing, fix the integration before the
  final take; do not substitute a static success label.
- Check the actual phone actions and signature verification before recording.
  The phone is an in-page response interface, not an external mobile push or a
  message to a real team. Do not imply either occurred.
- Speak the camera action only when the physical action is visible. A software
  crop is a close-up, not proof that the camera moved.
- Keep any recorded detection-to-camera sequence at original speed. Present-tense
  narration describes the captured action; it must not claim a recording is live.
- Keep public artifacts free of private scene images. Verify repository visibility
  before promising viewers access. Publishing the repo is a separate action.

## Claim boundaries and optional judge follow-up

The current runtime lets the Verifier decide alert eligibility. Analyst wording
is context, not an independent agreement gate. It currently sends a masked object
crop to the vision models; a confirmed alert can retain a source-frame face crop
for human review. Do not claim separate training datasets, parallel agreement,
or that the models receive the full scene unless that implementation changes and
is verified. Original scene and object evidence are retained for rejected claims;
a separate face crop is retained only for confirmed alerts.

The checkpoint is
`RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8`.
FP8 reduces bits per quantized weight versus BF16; it does not halve parameter
count or guarantee identical quality. The previous NVFP4 benchmark is not an
FP8 benchmark. Avoid blanket sub-two-second, accuracy, cost, or bandwidth claims.
A measured camera response includes inference, transport and physical settling.

vLLM supplies the serving API and supports continuous batching and paged KV-cache
management. The deployment enables automatic prefix caching for repeated Director
instructions; the existing NVFP4 A/B found no measurable gain. Do not describe it
as a measured acceleration. LLM Compressor produced the published checkpoint;
Sherlock does not run quantization during an investigation. Speculative decoding,
vLLM Omni and disaggregated serving are ecosystem options, not enabled features
of this demo. The local application signing key establishes record integrity,
not independent third-party attestation.

Sources: [Red Hat AI model card](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8),
[LLM Compressor](https://github.com/vllm-project/llm-compressor),
[vLLM](https://docs.vllm.ai/), and this repository's README benchmark tables.
