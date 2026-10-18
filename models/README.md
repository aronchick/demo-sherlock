# Scout checkpoint

`scout-r2-960.pt` is the default local detector. It is a YOLOv8n checkpoint
trained for 30 epochs at 960 pixels on the supplied room captures and the
training set's reference images. Runtime inference uses 640 pixels,
weapon confidence 0.15, and NMS IoU 0.6. Other classes retain confidence 0.35.

SHA-256:
`59ef75726532cd5c403e3e74b0a15cc8a789c71205bee89562cbe18cffd94cfe`

The original training output remains untouched. This copy is committed so a
fresh checkout does not depend on a temporary training directory. `/mode`
reports the loaded checkpoint hash; presenter reuse also checks model bytes.

See [the evaluation](../docs/SCOUT_R2_VALIDATION.md) for all 608 supplied
images, held-out results, remaining misses, false positives, latency, and
regressions on general reference images. The checkpoint is specialized for
this demo room; it has not been validated as a production security detector.

To return to the stock detector for comparison:

```sh
just down-local
YOLO_MODEL=yolov8n.pt SCOUT_WEAPON_CONF=0.35 \
  SCOUT_IMGSZ=640 SCOUT_NMS_IOU=0.7 just up
```

Remove those overrides and run `just down-local` followed by `just up` to
restore the project's configured checkpoint.
