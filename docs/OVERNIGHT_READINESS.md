# Recording readiness audit

Target: a three-minute Sherlock demonstration, reviewed at 1920×1080,
with five briefing slides and at least ten complete rehearsals.

## Acceptance evidence

| Requirement | Proof required | Current state |
| --- | --- | --- |
| Fast camera loop | Real YOLO detection, Brev tool call, physical OBSBOT response with timestamps | Fresh OBSBOT connection works; image nearly black, physical prop run pending |
| Phone response | Open notice, inspect location and pictures, investigate, notify, resolve | Investigate, paging and Notify team verified on ten full photo walkthroughs; building location metadata not implemented |
| Evidence | Summary modal, detailed timeline, JSON, image hashes and signed revisions verified | Saved photo incidents pass signature, artifact and revision-history checks |
| Expanso | Cloud-connected edge and core, active jobs, same incident received on core with matching crop hash | Verified real Cloud execution, remote disk-readback crop receipts and serving metrics on Nebius |
| vLLM ecosystem | Correct model names and serving proof; sourced FP8/LLM Compressor explanation | Both FP8 Director and SmolVLM passed inference smoke; narration and five slides updated |
| Three-minute story | Timed script, slides, complete visible flow | 350-word script with 3:00 screen cues; physical recording still pending |
| Ten rehearsals | Ten individually recorded outcomes and inspected browser evidence | Ten final full photo walkthroughs passed after fixes, zero browser errors and matching core receipts; negative cases also exercised |
| Recording layout | No clipping/overlap/internal scroll at 1920×1080 and 1280×720 | 720p board, phone and expanded custody pass overflow checks; JSON coloring and zero browser errors verified; ten final full walkthroughs passed |
| Startup/shutdown | Documented commands and truthful GPU billing state | `just down` reached Nebius STOPPED and closed local ports; restart passed both model smokes; Scaleway retained |

## Boundaries

The current runtime gives the Verifier the alert decision; the Analyst adds
a description after the Verifier check. It is not a two-model consensus. Only the masked object crop
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

Completed migration gates: both model smoke calls, Cloud core connected,
matching downstream incident/crop hashes, and verified Nebius stop.
Restart recovery passed both model inference checks, and repeated startup preserved
healthy Cloud executions. Ten final full walkthroughs passed after restart. The physical camera loop
remains pending before old-instance deletion readback.

## Retrained Scout and integration evidence

Both training runs completed without intervention. The selected checkpoint is
committed at `models/scout-r2-960.pt`; see `models/README.md` and
`docs/SCOUT_R2_VALIDATION.md` for the SHA-256, settings, rollback and limitations.
All 608 supplied images were evaluated. The calibrated held-out result is
35/40 weapon boxes found, four false-positive boxes and one triggered negative
frame among 80. General reference-image performance regressed; the report
does not claim broader production readiness.

`results/photo_integration/` retains ten full walkthrough records. Every take
used real YOLO and Nebius inference, software PTZ over a supplied photo,
the phone response controls, verified signed custody and an Expanso core
disk receipt. These are not ten physical-camera measurements or recorded
three-minute videos. Seven additional positive input cases alerted; an empty
hand false positive was rejected, another hand produced no trigger, and Coke
was ignored locally. The corresponding nine incident histories and image
hashes verified; all eight cases with transmitted crops received core receipts.

Browser review found a missing asset allowlist entry and missing saved-state
restoration after reload. Both fixes passed final browser validation on the restarted application.
`results/photo_integration/final-walkthroughs/summary.json` records the ten
passing retests; private screenshots remain in `.runtime/recording-validation/`.
A subsequent browser retest also confirmed that new events replace the saved
incident label with Current investigation.

## Final validation checkpoint

The integrated local gate passed 585 tests with one Ray warning and zero Ruff
findings. Twenty view tests passed again after narration corrections. The runtime
reports the selected Scout SHA, and signed incident metadata now records it.
Director prefetch configuration passed both model smoke calls; this warm-cache
reconfiguration is not a measured improvement in cold provider restart time.

The three changed standalone browser modules passed anti-slop lint and TypeScript
checking with browser-only libraries (no inherited unrelated Node or React types).
The final label correction passed 23 targeted tests and a browser event retest.

## Additional cold-cycle findings

The final cold-cycle test on October 18 exposed two startup problems after
`just down` reported STOPPED in 54.7 seconds:

- The provider continued rejecting starts with `status=STOPPING` beyond the
  original three-minute retry window. The bounded window is now ten minutes.
  A later attempt failed before that limit; its exact provider error was not
  retained. A subsequent explicit start succeeded.
- Brev omitted the not-yet-running instance from SSH configuration. Once Brev
  reported RUNNING/HEALTHY, refreshing restored the SSH alias. Readiness polling
  now refreshes on failures, at most every 30 seconds, within its ten-minute limit.

Both fixes are merged (#122 and #123). The full local gate passed 585 tests.
This recovery required intervention; do not describe it as an unattended clean
cold-start pass or promise a fast cold restart. Use `just down-local` for a brief
recording break when retaining GPU compute billing is acceptable.

The recovered cold start passed Director and VLM inference on October 18 at
05:40 UTC. The Director readiness loop took about 9.5 minutes after model
startup, in addition to provider shutdown/boot and the SSH-alias recovery.
Prefetch is supported and active; this run does not establish a startup speedup.
The old Scaleway instance remains retained and billable pending a visible physical
prop run. No old-instance deletion occurred.

After recovery, `/mode` confirmed the promoted Scout hash and both model endpoints,
OBSBOT was available, and all five Expanso Cloud jobs were running. `just down`
then closed the owned local services and Brev reported Nebius STOPPED. Provider
shutdown can lag that inventory state; retained storage remains billable.
The photo servers and isolated validation browser are closed. Claude's process
and the original training outputs were left untouched.
