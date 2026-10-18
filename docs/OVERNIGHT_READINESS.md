# Recording readiness audit

Target: a three-minute Sherlock demonstration, reviewed at 1920×1080,
with five briefing slides and at least ten complete rehearsals.

## Acceptance evidence

| Requirement | Proof required | Current state |
| --- | --- | --- |
| Fast camera loop | Real YOLO detection, Brev tool call, physical OBSBOT response with timestamps | USB pan, tilt and zoom target readback passed in the dark; visible detection-to-movement prop run pending |
| Phone response | Open notice, inspect location and pictures, investigate, notify, resolve | Investigate, paging and Notify team verified on ten full photo walkthroughs; location configuration implemented, actual labels awaited |
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
the current evidence policy.

This follows the deliberate change in merged
[PR #103](https://github.com/aronchick/demo-sherlock/pull/103), which replaced
the parallel scene-context implementation from #83. Its recorded real-model
evaluation found that the Analyst's structured verdict failed to parse in all
12 cases and supported all six cups as well as all six knives. Contact sheets
also weakened Verifier margins. The replacement reported 14/14 knives alerted
and 15/15 cups rejected using masked object crops, with the Verifier first.
Those are the PR's historical measurements, not a fresh GPU evaluation.

Preserve this tested decision for the recording. Parallel scene/person analysis
would require a new model evaluation before reintroduction; it is not an
unfinished feature to restore automatically. Narration must describe the
Verifier deciding and the Analyst describing, without claiming consensus.

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

## Dark-room validation after restart

The user confirmed that the black camera view is expected because the room is
dark. Hardware control was tested separately from photo-driven detection.
`results/recording_readiness/dark-room-validation.json` retains the results.

- OBSBOT USB readback reached the requested pan, tilt and zoom targets, then
  returned to pan 0, tilt 0, zoom 1. Pan and tilt settled in about 0.53 seconds
  each; zooming to 2x took 2.61 seconds. These are control-test timings, not
  detection-to-zoom latency or visual confirmation of the scene.
- A new cold start recovered SSH automatically but subsequently lost its SSH
  connection during Director loading. The containers continued running. A
  later `just up` reused them and passed both inference checks. Another warm
  startup completed in 61.14 seconds and reused the active Cloud pipelines.
  Unattended cold-start reliability is still unproven.
- A fresh supplied-photo incident, `20261018-041003-001`, produced a verified
  alert in 4.56 seconds. Signature, artifacts and revision history verified.
  Remote Expanso core disk readback matched the local weapon-crop hash. This
  used software PTZ over a photo, separately from the USB control test.
- Both the presenter checkout and the capture checkout now contain the same
  permanent Scout checkpoint and defaults. Its SHA matches the untouched
  `runs/scout_r2_960/weights/best.pt`, and the actual presenter reported that
  SHA at `/mode`. The leftover photo-rehearsal process was stopped.
- `just down` completed and Brev reported Nebius STOPPED. Local ports 8000,
  8020, 18001, 18002 and 23121 had no listeners. Retained storage remains
  billable, and provider shutdown can lag the inventory status.

The requested visible prop-to-camera loop remains a daylight validation step.
The old Scaleway instance has not been deleted; its compute billing continues.

The setup command now handles an SSH exit 255 with one reconnect and a
verification-only pass. That pass requires the exact expected containers to be
running, then checks model readiness and inference without starting, stopping,
replacing or pulling containers. Ordinary setup errors still fail immediately;
a second SSH failure is reported rather than retried indefinitely. Automated
tests cover those boundaries.

## Cold-start retest after #132

One cold start completed without intervention in **21 minutes 12 seconds**,
including both inference smokes, all five running Cloud jobs and presenter
readiness. The runtime reported the promoted Scout SHA. This run did not lose
SSH during model setup, so it does not exercise that recovery branch live.

The immediate second `just up` failed when SSH closed during source upload,
before setup ran. Upload now retries once after reconnecting, resending the
same committed archive bytes. Setup still waits for a complete transfer;
ordinary transfer errors and a second disconnect remain failures. Presenter
reuse needs a new runtime retest after this correction.

An earlier attempt was refused by the H100 guard before boot. Two subsequent
inventory reads identified the expected H100. The refused GPU field was not
retained, so its cause is unresolved; the error now includes the reported GPU
and instance type without weakening the guard.

Automatic `just down` completed in 60.88 seconds. Brev reported Nebius STOPPED,
and the owned demo ports were closed. See
`results/recording_readiness/cold-cycle-132.json` for the retained results.
Allow at least 25 minutes before recording for a cold start; this is planning
headroom around one measured run, not a startup guarantee. The capture checkout
has current code and weights, but use the configured presenter checkout at
`~/code/second-brain/projects/demos/demo-sherlock` for the camera and Cloud run.

## Startup and reuse verified after #133

The next cold start completed unattended in **20 minutes 6 seconds**. SSH
disconnected during source upload; the new retry reconnected and transferred
the archive successfully. Both models passed inference smoke tests, all five
Expanso Cloud jobs reported running, and the OBSBOT presenter became ready.

A second `just up` completed in **55.82 seconds**, preserving the presenter
PID and full ownership identity. It reused the running model containers and
Cloud executions. The presenter reported the promoted Scout SHA and received
fresh serving metrics for both remote models.

`just down` completed in **64.40 seconds**. Brev reported Nebius STOPPED, and
ports 8000, 8020, 18001, 18002 and 23121 were closed. Retained storage remains
billable. Scaleway remained RUNNING; deletion still awaits the agreed visible
physical-camera validation. No physical prop run occurred in this cycle.

`results/recording_readiness/startup-reuse-133.json` retains the timing,
runtime readback, presenter identity comparison and hashes of the private logs.
The earlier failed reuse attempt remains documented above; this later test
verifies the corrected behavior. The local gate passed 616 tests and Ruff,
with one existing Ray warning.

## Publication audit

The repository remains private. Gitleaks scanned 449 commits and reported no
credential findings, but that scan does not establish image privacy or rights.
The current tracked image set consists of illustrated UI captures and the
documented NASA fixture. No room capture dataset is tracked in the current tree.
The bundled Scout now includes its upstream AGPL license notice.

Historical commits reachable from `main` still contain `photos/IMG_0105.jpeg`
through `photos/IMG_0110.jpeg` and `IMG_0112.jpeg`. These show hackathon slides
and props; one slide photo carries an explicit confidentiality marking and
includes an attendee. Making the existing repository public would expose that
history. Do not treat removal from the current tree as removal from Git history.

A source-only archive of revision `7b0d6e0` was prepared locally, without Git
history, runtime files or those historical photos. Its SHA-256 is
`ba4d280231b5813107d66a228f77bf87a78c30e8714d15ea185fc83afe1565f3`.
The proposed publication is a separate public repository containing the clean
snapshot, preserving this private repository. Repository creation and the final
public name await the owner's confirmation. No visibility change, history
rewrite, new remote repository or public publication has occurred.
