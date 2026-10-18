# Live knife escalation, 2026-10-17

The attached OBSBOT's knife image scored 0.148 in YOLOv8n, below the normal
0.35 detection cutoff. The live configuration now uses `SCOUT_WEAPON_CONF=0.10`
for weapon candidates and keeps other classes at 0.35. This is a candidate
threshold, not confirmation: the Verifier still gates phone alerts. Lowering
it increases candidate investigations, including rejected scissors detections.

Tracking now matches both label and box, so a knife cannot inherit an already
reported remote or phone track. Expired tracks are removed before matching.
Weapon candidates use a 120 ms dwell, bounded by the configured dwell.
Ordinary objects are handled locally rather than calling the cloud Director.
Observed ordinary-object handling fell from roughly 500 ms to 7 ms.

The live preview reads fresh frames while evidence capture waits for the
gimbal. The Director requests one object close-up; face evidence is retained
from the original scene after confirmation, without a second camera move.
The local recording configuration caps OBSBOT zoom at 2× to reduce travel.

Live incident `20261017-190405-006` followed the real camera and Brev path:

- YOLO emitted a knife candidate.
- Director response: 348 ms; camera `look_at`: 276 ms.
- Object snapshot: 127 ms, with the camera already near its requested zoom.
- Verifier confirmed; Analyst described a knife.
- Completed report: 2222.6 ms.
- Phone delivery record: `delivered`; the knife notice opened in the phone UI.
- Image hashes, signature and signed history all verified.

This is one measured run, not a latency guarantee. Before the 2× cap,
large zoom changes took 6–9 seconds end to end. A settled camera image still
depends on physical movement, and a moving object can leave the crop before
capture. Rejected and inconclusive cases remain in evidence without a
confirmed phone alert.
