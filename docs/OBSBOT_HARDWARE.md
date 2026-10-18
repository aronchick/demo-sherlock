# OBSBOT Tiny 2 Lite hardware validation

Tested on macOS on 2026-10-17 with a directly connected Tiny 2 Lite,
USB `3564:fef9`, video name `OBSBOT Tiny 2 Lite StreamCamera`.
Interface 0, camera terminal 1 and extension unit 2 answered control reads.
Video frames were 1920×1080. Reported ranges were pan ±130°, tilt ±90°,
and raw zoom 0–100.

The hardware run found three issues that fake-transport tests could not expose:

- Positive UVC pan turns left. The adapter now reverses pan by default.
- A 1× to 2× zoom takes about two seconds. The capture wait now allows
  2.2 seconds per zoom-factor change, plus the existing stabilization wait.
- A combined-axis move can leave a stale pan value on the next command.
  The USB transport now reads the position and retries after 500 ms,
  failing after 1.7 seconds if it cannot confirm the target. Readback uses
  the existing 2° tolerance because fractional negative angles truncate
  toward zero: a −14.49° request reported −13°.

The probe now waits for all three axes to reach the requested target and
returns an error on timeout. It restores home before releasing the camera,
including when a check fails.

## Results

The corrected probe completed pan right, tilt up, 2× zoom and return-to-home.
Visual inspection confirmed direction and magnification. Pan +10° settled
in 0.55 seconds, tilt +5° in 0.54 seconds, the combined return-to-center and
2× zoom in 2.52 seconds, and zoom back to 1× in 1.58 seconds. The five-second
idle check observed zero drift; no controlled walking challenge was performed.

The application's `ToolBox.call('look_at', ...)` was tested against the
physical camera with a box in the right half of the full frame. The call
completed in 441 ms and a settled close-up was available in 3.141 seconds.
The dashboard `/cameras` response reported `backend: obsbot`,
`ptz: obsbot-usb`, and no warning. Final readback was pan 0, tilt 0, zoom 0
(1×). Scene images were inspected locally and are not part of this repository.

These measurements cover the camera tool and capture. They exclude YOLO,
Brev inference and delivery. They do not support an under-two-second claim
for the complete detection-to-settled-image loop. Tiny 2 and Tiny SE hardware
remain untested; calibration can be overridden through `OBSBOT_*` settings.
