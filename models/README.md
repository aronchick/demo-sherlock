# Scout checkpoint

`scout-r2-960.pt` is the default local detector. It is a YOLOv8n checkpoint
trained for 30 epochs at 960 pixels on the supplied room captures and the
training set's reference images. Runtime inference uses 640 pixels,
weapon confidence 0.15, and NMS IoU 0.6. Other classes retain confidence 0.35.

SHA-256:
`45db080dd23c079f61398253771ac735c93d2ddf4946cce58fa737f8b42f4e31`

The distribution archive removes seven training-machine path and Git metadata
strings. All 359 tensor entries and every other ZIP member are byte-for-byte
identical to the evaluated source archive, SHA-256 `59ef75726532cd5c403e3e74b0a15cc8a789c71205bee89562cbe18cffd94cfe`.
Historical evaluation records retain that source hash. `/mode` reports the
loaded distribution hash; presenter reuse also checks model bytes.

See [the evaluation](../docs/SCOUT_R2_VALIDATION.md) for all 608 supplied
images, held-out results, remaining misses, false positives, latency, and
regressions on general reference images. The checkpoint is specialized for
this demo room; it has not been validated as a production security detector.

## License

This checkpoint derives from Ultralytics YOLOv8n and is distributed under
[AGPL-3.0](LICENSE-ultralytics), consistent with the default model card in
`scripts/publish_scout_hf.py`. The repository's MIT license does not relicense
these weights or Ultralytics. See [Ultralytics licensing](https://www.ultralytics.com/license)
for its upstream terms and enterprise option. The included license text comes
from the installed `ultralytics` 8.4.180 package.

Training and evaluation code lives in `scripts/train_scout.py`,
`scripts/scout_data.py` and `scripts/eval_scout.py`. The source room photographs
remain local and are not part of this repository.

## Compare with the stock detector

To return to the stock detector for comparison:

```sh
just down-local
YOLO_MODEL=yolov8n.pt SCOUT_WEAPON_CONF=0.35 \
  SCOUT_IMGSZ=640 SCOUT_NMS_IOU=0.7 just up
```

Remove those overrides and run `just down-local` followed by `just up` to
restore the project's configured checkpoint.
