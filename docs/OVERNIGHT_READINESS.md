# Recording readiness audit

Target: a three-minute Sherlock demonstration, reviewed at 1920×1080,
with five briefing slides and at least ten complete rehearsals.

## Acceptance evidence

| Requirement | Proof required | Current state |
| --- | --- | --- |
| Fast camera loop | Real YOLO detection, Brev tool call, physical OBSBOT response with timestamps | Prior incident measured; fresh run pending |
| Phone response | Open notice, inspect location and pictures, investigate, notify, resolve | Overnight UI inspection underway |
| Evidence | Summary modal, detailed timeline, JSON, image hashes and signed revisions verified | Fresh interaction audit pending |
| Expanso | Cloud-connected edge and core, active jobs, same incident received on core with matching crop hash | Edge restart underway; missing tunnel and remote source packaging found |
| vLLM ecosystem | Correct model names and serving proof; sourced FP8/LLM Compressor explanation | Existing source available; narration revision underway |
| Three-minute story | Timed script, slides, complete visible flow | Revision underway |
| Ten rehearsals | Ten individually recorded outcomes and inspected browser evidence | 0 completed |
| Recording layout | No clipping/overlap/internal scroll at 1920×1080 and 1280×720 | Initial board screenshot captured |
| Startup/shutdown | Documented commands and truthful GPU billing state | Current Scaleway instance cannot stop; remains allocated |

## Boundaries

The current runtime gives the Verifier the alert decision; the Analyst describes
confirmed evidence. It is not a two-model consensus. Only the masked object crop
reaches the models. Original scene and face evidence are retained locally under
the current evidence policy. These differ from earlier requested parallel/context
behavior and must not be narrated as implemented.

Expanso transports events, incident records, masked image bytes, and serving
metrics. It does not execute the camera inference loop. Local HTTP acceptance
is not proof that the core persisted a record.

A rehearsal using generated camera scenes is UI coverage, not a real hardware
or GPU measurement. A captured incident can support evidence/phone interactions;
it cannot establish a new camera response time.

## Initial findings

- Brev model tunnel lacked the forward core-intake port and reverse metrics port.
- Brev source archive omitted jobs and edge node configuration.
- Expanso edge was disconnected and core jobs were queued.
- Recording script targeted four minutes and contained stale readiness notes.
- Existing H100 rejects stop; no instance has been deleted for this task.

## Approved infrastructure migration

The user authorized a stoppable Nebius 1× H100 at $5.40/hour plus storage,
with stop/start and complete demo verification before deleting Scaleway.

- Replacement: `sherlock-nebius-h100`, ID `hmbypnd6s`.
- Selected type: `gpu-h100-sxm.1gpu-16vcpu-200gb`, 80 GB VRAM;
  Brev search reports `stoppable: true`. Requested disk: 200 GB.
- Old instance: `sherlock-h100`, ID `ctuant3yw`; do not delete before all gates pass.
- Private backups are retained in `.runtime/migration/` (gitignored):
  `scaleway-project-20261018.tar.gz` and `scaleway-original-20261018.tar.gz`.
  Both archives passed full listing validation. The latter includes the old
  `~/sherlock/results` directory; virtual environments and Git internals were excluded.
- SHA256 project archive:
  `94a1ba651e438a58286bd1ebbb921cbe19eb691804c19fe8a397736a1767d60e`.
- SHA256 original archive:
  `eb56681a1da18ce4c346d7dc3256ec5edfd5c3ea88b4b9e63bc6bcc4d1dfa9df`.

Pending gates: both model smoke calls, Cloud core connected, matching downstream
incident/crop hash, physical camera loop, ten complete rehearsals, idempotent
startup, stop/start/recovery, then old-instance deletion readback.
