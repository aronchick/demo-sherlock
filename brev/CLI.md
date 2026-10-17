# Run Sherlock on the existing Brev H100 from your Mac

The checked instance is `sherlock-h100` (H100 PCIe 80GB). Run these commands from
the local `demo-sherlock` checkout. You do not need an interactive SSH session.
The Brev CLI must already be logged in; `brev ls --json` shows your instances.

## 1. Configure or reuse the GPU services

```bash
just brev-up sherlock-h100
```

This starts the named instance if stopped, waits for SSH, and copies only selected
committed source files into a revision-specific release directory. It checks GPU
and Docker availability and installs uv 0.12.17 in that release if needed.
Existing running containers with the exact FP8 Director and SmolVLM are reused.
Otherwise it validates ownership of both named containers before replacing them.
It preserves the Hugging Face download cache and every other container.

New model containers bind to GPU-host loopback. Reused containers retain their
existing binding configuration. The command runs the Director tool-call and VLM
smoke checks with exact model IDs. It does not create or delete a GPU instance.
If Docker or GPU support is missing, it stops with a concrete error rather than
changing host drivers. The existing Brev H100 already has those dependencies.

## 2. Connect and keep this terminal open

```bash
just brev-connect sherlock-h100
```

This forwards local 18001 and 18002 to remote 8001 and 8002, then repeats inference
checks through the tunnel. It refuses occupied ports and writes its connection
receipt only after both checks pass. Your `.env` and local model servers are
unchanged. Ctrl-C closes this tunnel and removes its receipt.

## 3. Start the cameras and recording UI in a second terminal

If the previous presenter was started with `just up`, stop it first:

```bash
just down
just brev-live
```

`brev-live` verifies the tunnel again, then runs the cameras and dashboard locally
with the FP8 endpoint overrides. Camera selection and Expanso settings still come
from your existing configuration. Only the presenter runs on localhost; this does
not set up or change the Expanso data plane.

For a camera-free rehearsal against the real remote models:

```bash
CAMERAS=hero:fake:synthetic,scout1:fake:synthetic \
  just brev-live --synthetic
```

The rehearsal does not establish YOLO or physical camera response time. Before
recording the opening, verify a real detection, Director tool call, and returned
close-up as one continuous sequence at original speed.

## Retire a different old machine, only when needed

There is currently only one listed machine; do not retire it to switch models.
If you later bring up a replacement, keep its verified tunnel open and run:

```bash
just brev-retire OLD_INSTANCE NEW_INSTANCE
```

This refuses identical instance IDs, requires the current tunnel to belong to the
replacement, reruns inference checks, then requests a stop of the named old
machine and reads its status back. It never deletes either instance. An
unconfirmed stop is reported as unfinished, not successful shutdown.

Read status at any time:

```bash
just brev-status
```

If you need to inspect a failure on the machine:

```bash
brev exec sherlock-h100 'docker logs --tail 100 director'
brev exec sherlock-h100 'docker logs --tail 100 vlm'
```
