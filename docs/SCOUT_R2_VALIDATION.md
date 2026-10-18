# Scout R2 validation

Recommend the **960-trained checkpoint at 640px inference**, weapon confidence
0.15 and NMS IoU 0.6 for the recording room. Direct inference on the supplied
capture images supports this choice; no training process, label, checkpoint or
runtime default was changed by this evaluation.

```dotenv
SCOUT_CONF=0.35
SCOUT_WEAPON_CONF=0.15
SCOUT_IMGSZ=640
SCOUT_NMS_IOU=0.6
```

Selected source checkpoint:
`/Users/daaronch/code/demo-sherlock-capture/runs/scout_r2_960/weights/best.pt`

SHA-256:
`59ef75726532cd5c403e3e74b0a15cc8a789c71205bee89562cbe18cffd94cfe`

The parent deployment must preserve this hash when installing a stable runtime
path. This report proves image-level detection results, not physical camera
movement, cloud verification, downstream delivery or the completed video.

## Results on the user's pictures

All **608 supplied capture images** were evaluated. The 122-image test subset
contains 42 knife-session frames and 80 other-prop/empty frames. Forty knife
frames have weapon boxes; two are occluded-camera frames with none.

| Model and population | Weapon TP / GT | Recall | FP boxes | Precision | Negative frames triggering |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock, test 122 | 1 / 40 | 2.5% | 0 | 100% | 0 / 80 |
| Selected, test 122 | 35 / 40 | 87.5% | 4 | 89.74% | 1 / 80 |
| Selected, all captures 608 | 189 / 200 | 94.5% | 27 | 87.5% | See population note |

Both test rows use confidence 0.15, 640px inference, NMS IoU 0.6 and
class-correct one-to-one matching at IoU >= 0.5. Stock's 100% precision is based
on just one correct prediction and does not offset its 39 misses. Weapon
mapping combines stock knife/scissors/baseball-bat classes and the fine-tuned
model's merged `weapon` class.

The all-capture row includes **426 training and 60 validation images** and must
not be described as held-out performance. Its 400 non-knife-session images
include one existing weapon annotation in a cap session; they are not all
negative ground truth. The 80 test negatives have no weapon annotations.

The selected model finds all eight test knives in the difficult vertical-blade
session `193421`. The 640-trained candidate found only four at confidence 0.15.
One remaining negative trigger is an empty-hand image from the second camera,
`room_none_scout1-20261017-171153_0037.jpg`, with confidence 0.367. The separate
Verifier must reject this claim; detection alone must not deliver an alert.

Five test knives still produce no weapon box:

- `room_knife_hero-20261017-173649_0034.jpg`
- `room_knife_hero-20261017-173649_0036.jpg`
- `room_knife_hero-20261017-184411_0032.jpg`
- `room_knife_hero-20261017-184449_0037.jpg`
- `room_knife_hero-20261017-184449_0038.jpg`

These are detector misses, not merely boxes falling short of matching IoU.
Do not inject detections for these frames during an end-to-end demonstration.

## Model and operating-point comparison

Both MPS training jobs finished their requested 30 epochs normally. The 640px
run completed in about 888 seconds, and the 960px run in about 1,729 seconds.
Both started from YOLOv8n, with early-stop patience 10. Evaluation began only
after each process terminated and optimizer stripping finished.

The alternate 640-trained checkpoint has SHA-256
`8e24786cb2593edca326a4b3d98291161d6b457107798dfb7d2457be5eaf9772`.

Before changing NMS, confidence 0.15 gives:

| Training size | Inference size | Test TP / 40 | FP boxes | Negative frames |
| ---: | ---: | ---: | ---: | ---: |
| 640 | 640 | 30 | 16 | 2 / 80 |
| 640 | 960 | 31 | 16 | 2 / 80 |
| 960 | 640 | 35 | 17 | 1 / 80 |
| 960 | 960 | 34 | 14 | 1 / 80 |

640px inference retains more test matches for the selected model and avoids
larger input processing. Directly passing NMS IoU 0.6 to YOLO then reduces its
17 false-positive boxes to **4**, while retaining all **35** true positives.
Many removed boxes were duplicate detections of one knife. This final result
was measured with actual YOLO inference, not just a cached post-filter.

Confidence sweeps covered 0.10, 0.15, 0.20, 0.25, 0.30 and 0.35. For the selected
model at 640px with default NMS, recall was respectively 87.5%, 87.5%, 85%,
82.5%, 77.5% and 75%. The 0.15 threshold removes low-confidence duplicates
without losing matches relative to 0.10. Normal non-weapon classes retain the
runtime's 0.35 threshold; the reported weapon metrics are unaffected by that
separate filter.

## Local latency

CPU single-image prediction, 640px input, confidence 0.15 and NMS IoU 0.6:

| Model | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Stock YOLOv8n | 20.84 ms | 24.16 ms | 29.96 ms |
| Selected checkpoint | 21.91 ms | 27.03 ms | 28.82 ms |

Each row has five warmups followed by 30 measured single-image calls on the
same ordered test images. Timing includes file read/decode and prediction,
excludes model loading, and uses CPU with `OMP_NUM_THREADS=2` and
`MKL_NUM_THREADS=2`. It is not a camera-to-alert latency measurement or a promise
about sustained throughput. The selected model did not introduce a large
single-image latency penalty in this check.

## General-image regression

The separate 144-image COCO test subset shows a tradeoff. At the final 0.15 /
640px / NMS 0.6 settings:

| Class | Stock precision / recall | Selected precision / recall |
| --- | ---: | ---: |
| Weapon | 68.75% / 44.44% | 45.62% / 36.87% |
| Person | 65.80% / 79.84% | 57.50% / 73.52% |
| Bottle | 45.45% / 59.38% | 32.51% / 41.25% |

This is a room-specific improvement with a general-image regression. Do not
claim that nothing else broke, or that this is a validated general-purpose
security detector. The badge class has no ground-truth support in this dataset.

## Provenance and limitations

The source is the capture checkout's `data/scout` dataset. The evaluation uses
`eval_scout.py` / `scout_data.py` prediction mapping and matching semantics;
the final NMS comparison passes the option directly to Ultralytics. Checkpoint
hashes were checked before and after the capture evaluation.

There are no exact image-hash duplicates across train/validation/test. However,
each capture session is split in time order: first 70% train, next 10% validation,
last 20% test, with no separation gap. The same person, room and prop occur in
all splits. These are correlated held-out frames, not new-session generalization.
Thresholds and NMS were explored on the test subset, so it also serves as a
calibration set. A fresh capture session is needed for an unbiased final test.

All room validation/test images have review markers, but a recent marker backup
is named `before-bulk-accept`. Markers alone do not establish that a human
checked every rectangle. The existing labels were scored unchanged; 387 room
training frames lack review markers. No unsupported labels were silently fixed
to improve results.

Evaluation JSON is retained in `results/scout_r2/`:

- `dataset-manifest.json`: all 608 image/label hashes and original splits.
- `stock-detail.json`: baseline capture predictions and threshold breakdown.
- `scout-r2-640-detail.json`: alternate candidate on all captures.
- `scout-r2-960-at640-detail.json` and `scout-r2-960-at960-detail.json`: selected
  candidate at both inference sizes, including all-capture threshold sweeps.
- `recommended-direct-nms.json`: final actual NMS predictions and session counts.
- `selected-full-test-and-latency.json`: stock/selected comparison on all 266 test
  images, separate COCO metrics and latency samples summarized above.
- `ten-integration-inputs.json`: ten exact capture/label paths spanning seven
  positives and three negatives, including thin blades and an empty-hand false
  positive. These are input cases, not proof that integration passed.

The older evaluation script headline hardcodes confidence 0.35 even when a
different threshold is supplied. This report uses actual JSON configuration
and inference settings, not that headline. Scene photographs remain local;
no private image is included in this document.
