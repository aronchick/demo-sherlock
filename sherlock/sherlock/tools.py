"""The Director's hands. Every tool returns a dict; errors come back as
{"error": ...} rather than raising, so the model gets to decide what to do."""
from __future__ import annotations

import base64
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .camera import Camera, CameraError
from .events import BUS
from .privacy import FaceMasker

SUBJECTS = ("weapon", "object", "face")
ACTIONS = ("log", "alert", "ignore")
_DEVICE_TOOLS = ("look_at", "move", "snapshot")


def _preview(role: str, img: np.ndarray, stage: str = "model_input", **metadata) -> None:
    """Publish a bounded copy of the actual model input before inference starts.

    This travels only on the dashboard bus. Face previews use evidence_preview, never model_input.
    PNG keeps the thumbnail faithful and limits the longest edge to 256 pixels.
    """
    height, width = img.shape[:2]
    scale = min(1, 256 / max(height, width))
    thumb = cv2.resize(img, (max(1, round(width * scale)), max(1, round(height * scale))))
    ok, encoded = cv2.imencode(".png", thumb)
    if ok:
        BUS.emit(stage, f"{role} " + ("human-review evidence" if stage == "evidence_preview" else
                 "reading the object crop"), role=role, **metadata,
                 image="data:image/png;base64," + base64.b64encode(encoded).decode("ascii"))

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "look_at",
        "description": "Point a camera at a box seen in its wide view and zoom so the box fills the frame. "
                       "Use the weapon bbox first, then the head bbox for the face evidence crop.",
        "parameters": {"type": "object", "properties": {
            "camera": {"type": "string"},
            "bbox": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4,
                     "description": "[x1, y1, x2, y2] pixels in the wide frame"}},
            "required": ["camera", "bbox"]}}},
    {"type": "function", "function": {
        "name": "move",
        "description": "Pan/tilt/zoom a camera directly. pan -140..140 deg, tilt -70..30 deg, zoom 1..4.",
        "parameters": {"type": "object", "properties": {
            "camera": {"type": "string"}, "pan": {"type": "number"},
            "tilt": {"type": "number"}, "zoom": {"type": "number"}},
            "required": ["camera"]}}},
    {"type": "function", "function": {
        "name": "snapshot",
        "description": "Capture what the camera sees right now and store it as evidence. subject=weapon or "
                       "object stores the model-safe crop for the parallel checks after acquisition. subject=face only "
                       "stores person context without invoking a model during acquisition. Never snapshot a benign object.",
        "parameters": {"type": "object", "properties": {
            "camera": {"type": "string"},
            "subject": {"type": "string", "enum": list(SUBJECTS),
                        "description": "weapon | object | face. Evidence id becomes <subject>_crop"},
            "question": {"type": "string",
                         "description": "What to ask the Analyst about the object. Never ask about a person"}},
            "required": ["camera", "subject"]}}},
    {"type": "function", "function": {
        "name": "verify",
        "description": "Second VLM pass: check a claim about the weapon against the weapon crop (never the "
                       "face crop). Call after the snapshots, before report.",
        "parameters": {"type": "object", "properties": {
            "claim": {"type": "string", "description": "One sentence: what the weapon is"}},
            "required": ["claim"]}}},
    {"type": "function", "function": {
        "name": "list_cameras",
        "description": "List cameras and their current pose/status. Use when a camera fails.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "report",
        "description": "Finish the investigation. Call this exactly once at the end. action=ignore for benign "
                       "objects, alert for a verified weapon, log otherwise.",
        "parameters": {"type": "object", "properties": {
            "claim": {"type": "string", "description": "One sentence: what the object is"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "action": {"type": "string", "enum": list(ACTIONS)},
            "evidence": {"type": "array", "items": {"type": "string"},
                         "description": "Evidence ids returned by snapshot, e.g. [\"weapon_crop\", \"face_crop\"]"}},
            "required": ["claim", "confidence", "action"]}}},
]


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def _gap(box, point) -> float:
    """Distance from a point to a box; 0 inside it."""
    x, y = point
    return float(np.hypot(max(box[0] - x, 0, x - box[2]), max(box[1] - y, 0, y - box[3])))


def _norm(claim: str) -> str:
    return " ".join(str(claim).lower().split()).rstrip(".")


def _short(v, limit: int = 40) -> str:
    """Keep each tool call on one log line; the full args are in the event data."""
    r = repr(v)
    return r if len(r) <= limit else r[:limit - 2] + "…" + r[-1]


def _box(v) -> tuple | None:
    """A usable [x1, y1, x2, y2], or None for whatever a confused model sent."""
    if isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(n, (int, float)) for n in v):
        return tuple(v)
    return None


# What a model sees of a weapon or object: the aimed box plus 20% of its own width and height
# per side, never the whole zoomed view. 20% is about twice a detector's typical edge error, so
# a box that clips the blade still yields the whole object, and it absorbs the hardware aim error
# view_box leaves to the caller. It stops short of neighbours: on 2026-10-17 the full view around
# an 86 px cup held a real knife's handle 23 px away, and the Verifier confirmed "a knife".
# Each side gets at least 10% of the longer side, so an upright knife is not cut to a sliver.
MODEL_CROP_MARGIN, MODEL_CROP_FLOOR = 0.2, 0.1


def _crop(img: np.ndarray, box: tuple, margin: float = 0.3, floor: float = 0.0) -> np.ndarray:
    """The box plus `margin` of its size on every side (at least `floor` of its longer side),
    clamped to the image."""
    h, w = img.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    least = max(bw, bh) * floor
    mx, my = max(bw * margin, least), max(bh * margin, least)
    x1, y1 = max(int(box[0] - mx), 0), max(int(box[1] - my), 0)
    x2, y2 = min(int(box[2] + mx), w), min(int(box[3] + my), h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return img  # box fell outside the view: keep the whole frame rather than nothing
    return img[y1:y2, x1:x2].copy()


# The chain the dashboard draws, in plain language. A call's step comes from what
# it does, not from how many calls came before it, so a retry keeps its number.
_SAYS = {1: "zooming {camera} to the {subject}", 2: "capturing the {subject} crop for parallel checks",
         3: "zooming {camera} to the head", 4: "storing the person-context crop",
         5: "verifying “{claim}”", 6: "reporting · {action}"}


class ToolBox:
    def __init__(self, cameras: dict[str, Camera], analyst, snapshot_dir: str = "snapshots", verifier=None,
                 masker=None):
        self.cameras = cameras
        self.analyst = analyst
        self.verifier = verifier
        # Loaded here, not on first use: a missing detector stops startup, not an investigation.
        self.masker = masker if masker is not None else FaceMasker()
        self.snapshot_dir = snapshot_dir
        self.last_crop: np.ndarray | None = None  # latest crop of any incident, for the live view
        self.injected_failures = 0  # demo switch: next N device calls fail
        self.injected_corruptions = 0  # demo switch: next N device calls get garbage arguments
        self.injected_disagreements = 0
        self.defer_analysis = False
        self.on_capture = None
        self.begin_incident()

    def _cam(self, name: str) -> Camera:
        if name not in self.cameras:
            raise KeyError(f"unknown camera {name!r}; known: {list(self.cameras)}")
        return self.cameras[name]

    def inject_failure(self, n: int = 1) -> None:
        self.injected_failures += n

    def inject_corruption(self, n: int = 1) -> None:
        self.injected_corruptions += n

    def begin_incident(self, detection=None, incident_id: str = "") -> None:
        """Fresh evidence per investigation, so an ignore can never inherit (and
        re-verify) the previous incident's crop."""
        self.incident = {"id": incident_id, "threat": getattr(detection, "threat", "unknown"),
                         "bbox": getattr(detection, "bbox", None), "head": getattr(detection, "head", None),
                         "label": getattr(detection, "label", "")}
        self._failed_steps: set[int] = set()  # chain steps that failed; a later success is a recovery
        self.provenance: dict[str, dict] = {}
        self.evidence: dict[str, np.ndarray] = {}
        self.evidence_captured_at: dict[str, float] = {}
        self.evidence_camera: dict[str, str] = {}  # which camera captured each crop (differs after recovery)
        self.descriptions: dict[str, str] = {}  # Analyst text per crop it saw
        self.masked: dict[str, tuple[int, float]] = {}  # model-bound crop -> (faces masked, ms)
        self.aim: dict[str, tuple] = {}  # camera -> box its last good look_at targeted; absent = unknown
        self.verdict: dict | None = None
        self._verified: tuple | None = None  # (claim, crop id, crop version) the verdict is about
        self._versions: dict[str, int] = {}
        self._shots = 0
        self.steps = 0

    def _chain(self, name: str, args: dict) -> dict:
        """Where a call sits in the six-step chain: step, subject and camera. Read
        from the arguments as sent, before any injected corruption."""
        step, subject, camera = None, None, args.get("camera")
        if name == "look_at":
            box = _box(args.get("bbox"))
            subject = "face" if box and self._near_head(box) else self._object_subject()
            step = 3 if subject == "face" else 1
        elif name == "snapshot":
            subject = args.get("subject", "object")
            step = 4 if subject == "face" else 2
        elif name == "verify":
            step, camera = 5, self.evidence_camera.get(self.model_crop()[0] or "")
        elif name == "report":
            step = 6
        return {"step": step, "subject": subject, "camera": camera if isinstance(camera, str) else None,
                "retry": step in self._failed_steps}

    def _object_subject(self) -> str:
        return "weapon" if self.incident["threat"] == "weapon" else "object"

    def call(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        self.steps += 1
        BUS.bump("tool_calls")
        chain = self._chain(name, args)
        tag = {"call": self.steps, "tool": name, "incident": self.incident["id"],
               "label": self.incident["label"], **chain}
        if chain["step"]:
            # The Director's state in plain language, before the call runs. Its own
            # stage, so views that animate every director event (/pipeline) skip it.
            say = _SAYS[chain["step"]].format(claim=args.get("claim", ""), action=args.get("action", ""),
                                              **{k: chain[k] for k in ("camera", "subject")})
            BUS.emit("narration", say, **tag,
                     **({"claim": args["claim"]} if isinstance(args.get("claim"), str) else {}))

        def failed(msg: str, **data) -> None:
            BUS.bump("tool_errors")
            if chain["step"]:
                self._failed_steps.add(chain["step"])
            BUS.emit("tool", msg, error=True, **data, **tag)

        if name in _DEVICE_TOOLS and self.injected_failures > 0:
            self.injected_failures -= 1
            if name != "snapshot":
                self.aim.pop(str(args.get("camera")), None)  # a failed move: pose unknown
            failed(f"{name}({args.get('camera')}) FAILED (injected)", data=args)
            return {"error": f"{name}: device returned garbage (simulated fault)"}
        if name in _DEVICE_TOOLS and self.injected_corruptions > 0:
            self.injected_corruptions -= 1
            args = {k: "\x00garbage" for k in args}  # what a confused model might send
            BUS.emit("tool", f"{name} args corrupted (injected)", **tag)
        if name == "snapshot" and self.incident["threat"] == "benign":
            BUS.emit("tool", "snapshot refused: benign object, no VLM call needed", error=True, **tag)
            return {"error": "benign object: no closer look needed. Call report with action='ignore'."}
        try:
            fn = getattr(self, f"t_{name}")
        except AttributeError:
            BUS.bump("tool_errors")
            return {"error": f"no such tool {name!r}"}
        try:
            out = fn(**args)
        except (TypeError, KeyError, ValueError, CameraError) as e:
            failed(f"{name} error: {e}", data=args)
            return {"error": str(e)}
        if chain["retry"]:
            self._failed_steps.discard(chain["step"])
            BUS.bump("recoveries")  # counted once the retry worked, not when it was attempted
        shown = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
        msg, masked = f"{name}({shown})", {}
        if name == "snapshot" and out.get("evidence_id") in self.masked:
            # On the event, not in the result: the Director has no use for a face count.
            n, ms = self.masked[out["evidence_id"]]
            masked = {"faces_masked": n, "mask_ms": round(ms, 1)}
            if n:
                msg += f" · {n} face{'s' if n > 1 else ''} masked"
        BUS.emit("tool", msg, (time.perf_counter() - t0) * 1000, result=out, **masked, **tag)
        return out

    # --- tools -------------------------------------------------------------
    def t_look_at(self, camera: str, bbox: list) -> dict:
        if len(bbox) != 4 or any(not isinstance(v, (int, float)) for v in bbox):
            raise ValueError("bbox must be [x1,y1,x2,y2] numbers")
        cam = self._cam(camera)
        self.aim.pop(camera, None)  # a move that fails halfway leaves the pose unknown
        box = tuple(int(v) for v in bbox)
        cam.look_at(box)  # each backend maps the box to its own optics
        self.aim[camera] = box
        return {"ok": True, **cam.describe()}

    def t_move(self, camera: str, pan=None, tilt=None, zoom=None) -> dict:
        cam = self._cam(camera)
        self.aim.pop(camera, None)  # raw degrees: nothing says what is in view now
        cam.move(pan, tilt, zoom)
        return {"ok": True, **cam.describe()}

    def aimed_at_face(self, camera: str) -> bool:
        """Could this camera's view be a person's head? Decided from where the camera
        was pointed, never from the subject the Director claims, so a mislabelled head
        zoom still can't reach a model. Unknown aim counts as a face whether or not
        this detection has a head box: the camera may still sit on the last incident's
        face zoom, so only a successful look_at in this incident clears it."""
        aim = self.aim.get(camera)
        return True if aim is None else self._near_head(aim)

    def _near_head(self, box: tuple) -> bool:
        """Is this box closer to the incident's head box than to its weapon box?"""
        head, weapon = self.incident["head"], self.incident["bbox"]
        if head is None or weapon is None:
            return False
        if _iou(box, weapon) >= 0.5:
            return False  # the weapon box itself, even when it is held up near the face
        centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        return _gap(head, centre) <= _gap(weapon, centre)

    def model_crop(self) -> tuple[str | None, np.ndarray | None]:
        """The crop a model may see: weapon (or object), never the face."""
        for eid in ("weapon_crop", "object_crop"):
            if eid in self.evidence:
                return eid, self.evidence[eid]
        return None, None

    def check_claim(self, claim: str) -> dict:
        eid, crop = self.model_crop()
        if crop is None:
            raise ValueError("nothing to verify: snapshot the weapon first")
        if self.verifier is None:
            raise ValueError("no verifier configured")
        _preview("verifier", crop)
        self.verdict = self.verifier.check(crop, claim)
        self._verified = (_norm(claim), eid, self._versions[eid])
        return self.verdict

    def verdict_for(self, claim: str) -> dict | None:
        """The verdict, only if it checked this claim against the crop as it is now."""
        eid, _ = self.model_crop()
        if self.verdict is None or eid is None or self._verified != (_norm(claim), eid, self._versions[eid]):
            return None
        return self.verdict

    def t_snapshot(self, camera: str, subject: str = "object", question: str = "What is this object?") -> dict:
        if subject not in SUBJECTS:
            raise ValueError(f"subject must be one of {list(SUBJECTS)}")
        cam = self._cam(camera)
        if subject != "face" and self.aimed_at_face(camera):
            raise ValueError(f"refused: {camera} is not aimed at the weapon, and a head view never goes to "
                             "a model. look_at the weapon bbox first, or use subject='face' for evidence")
        img = cam.snapshot()
        captured_at = time.time()
        aim = self.aim.get(camera)
        if aim is not None:
            box = cam.view_box(aim, (img.shape[1], img.shape[0]))
            if subject == "face":
                # A head crop, not the chest-up view the zoom leaves, so the review crop
                # holds the person's head and nothing the weapon chain needed.
                img = _crop(img, box)
            else:
                # The zoom keeps the frame's aspect ratio, so its view runs well past a narrow
                # box: neighbouring objects, or the face above a raised knife.
                img = _crop(img, box, MODEL_CROP_MARGIN, MODEL_CROP_FLOOR)
        eid = f"{subject}_crop"
        if subject != "face":
            # The second line behind the aim rules: a face the box still caught, such as a
            # seated child's mouth beside the knife, is filled solid. The stored crop is the
            # masked one, so the Verifier and the saved evidence see exactly what the Analyst saw.
            t1 = time.perf_counter()
            try:
                img, n = self.masker(img)
            except cv2.error as e:
                raise ValueError(f"face masking failed, so the crop went to no model: {e}") from e
            self.masked[eid] = (n, (time.perf_counter() - t1) * 1000)
        self.last_crop = img
        self.evidence[eid] = img
        self.evidence_camera[eid] = camera
        self.evidence_captured_at[eid] = captured_at
        self.provenance[eid] = {"camera": camera, "captured_at": captured_at,
            "source_artifact": "original_frame" if "original_frame" in self.evidence else None,
            "relationship": "subsequent_capture", "bbox": list(aim) if aim else None}
        if self.on_capture:
            self.on_capture()
        if subject == "face" and self.defer_analysis:
            _preview("face", img, "evidence_preview", incident=self.incident["id"])
        self._shots += 1
        self._versions[eid] = self._shots  # a new crop invalidates any verdict on the old one
        if subject == "face":
            # Acquisition only. The parallel investigation later includes it in labeled scene context.
            return {"evidence_id": eid, "stored": True, "analyzed": False}
        if self.defer_analysis:
            return {"evidence_id": eid, "stored": True, "description": "", "analysis": "pending"}
        BUS.bump("vlm_calls")
        _preview("analyst", img)
        out = self.analyst.describe(img, question)
        self.descriptions[eid] = out.get("description", "")
        return {**out, "evidence_id": eid}

    def t_verify(self, claim: str) -> dict:
        if self.defer_analysis:
            return {"supported": False, "reason": "parallel checks pending"}
        return self.check_claim(claim)

    def build_model_context(self) -> np.ndarray | None:
        """One labeled scene/object/person image for both independently prompted checks."""
        eid, crop = self.model_crop()
        if crop is None:
            return None
        names = [name for name in ("original_frame", eid, "face_crop") if name in self.evidence]
        context = np.full((752, 768, 3), 28, dtype=np.uint8)
        regions = {"original_frame": (0, 0, 768, 448), eid: (0, 448, 384, 304),
                   "face_crop": (384, 448, 384, 304)}
        for name in ("original_frame", eid, "face_crop"):
            x, y, width, height = regions[name]
            img = self.evidence.get(name)
            cv2.putText(context, name.replace("_", " "), (x + 8, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            if img is None:
                cv2.putText(context, "NOT CAPTURED", (x + 8, y + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
                continue
            provenance = self.provenance.get(name, {})
            camera = self.evidence_camera.get(name, "unknown")
            captured = self.evidence_captured_at.get(name)
            stamp = f"{captured:.3f}" if isinstance(captured, (int, float)) else "unknown time"
            cv2.putText(context, f"{camera} / {stamp}", (x + 8, y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1)
            cv2.putText(context, provenance.get("relationship", "unknown provenance"), (x + 8, y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1)
            scale = min((width - 16) / img.shape[1], (height - 64) / img.shape[0])
            tile = cv2.resize(img, (max(1, round(img.shape[1] * scale)), max(1, round(img.shape[0] * scale))))
            left = x + (width - tile.shape[1]) // 2
            top = y + 58 + (height - 64 - tile.shape[0]) // 2
            context[top:top + tile.shape[0], left:left + tile.shape[1]] = tile
        self.evidence["model_context"] = context
        self.evidence_camera["model_context"] = self.evidence_camera.get(eid, "unknown")
        self.evidence_captured_at["model_context"] = time.time()
        self.provenance["model_context"] = {
            "camera": self.evidence_camera["model_context"],
            "captured_at": self.evidence_captured_at["model_context"],
            "source_artifact": None, "relationship": "composite", "sources": names,
            "bbox": None, "missing_sources": [n for n in ("original_frame", eid, "face_crop") if n not in names]}
        if self.on_capture:
            self.on_capture()
        return context

    def assess_parallel(self, claim: str) -> dict:
        """Independent prompts on one immutable scene/object/person context, then comparison."""
        eid, _ = self.model_crop()
        image = self.build_model_context()
        missing = {"status": "uncertain", "reason": "No object crop available",
                   "analyst_supports": None, "verifier_supports": None, "claim": claim}
        if image is None:
            BUS.emit("comparison", missing["reason"], **missing)
            return missing
        image = image.copy()
        image.flags.writeable = False
        version = self._versions[eid]
        custody = BUS.captured_events()

        def run(role):
            with BUS.capture(custody):
                _preview(role, image, incident=self.incident["id"], evidence="model_context")
                try:
                    if role == "analyst":
                        BUS.bump("vlm_calls")
                        return self.analyst.assess(image, claim)
                    return self.verifier.check(image, claim)
                except Exception as exc:  # noqa: BLE001 - endpoint failure must retain the evidence
                    BUS.emit(role, f"Assessment failed: {type(exc).__name__}")
                    return {"error": type(exc).__name__}

        with ThreadPoolExecutor(max_workers=2) as pool:
            analyst = pool.submit(run, "analyst")
            verifier = pool.submit(run, "verifier")
            a, v = analyst.result(), verifier.result()
        if self.injected_disagreements:
            self.injected_disagreements -= 1
            a = {"description": "A ceramic cup. Simulated disagreement.", "claim": claim, "supported": False}
            BUS.emit("analyst", "SIMULATED disagreement injected: a ceramic cup", simulated=True)
        description = str(a.get("description", ""))
        self.descriptions[eid] = description
        self.verdict = v
        self._verified = (_norm(claim), eid, version)
        supports = a.get("supported")
        if type(supports) is not bool or a.get("claim") != claim or "error" in a:
            supports = None
        checked = v.get("supported") if "error" not in v and v.get("determinate", True) else None
        if self._versions.get(eid) != version:
            supports = checked = None
        status = "uncertain" if supports is None or checked is None else (
            "agreed" if supports == checked else "disagreed")
        reason = ("Both checks support the claim" if supports is True and checked is True else
                  "Both checks reject the claim" if supports is False and checked is False else
                  "Checks disagree" if status == "disagreed" else "At least one check is inconclusive")
        result = {"status": status, "reason": reason, "analyst_supports": supports,
                  "verifier_supports": checked, "claim": claim}
        BUS.emit("comparison", reason, **result)
        return result

    def t_list_cameras(self) -> dict:
        return {"cameras": [c.describe() for c in self.cameras.values()]}

    def t_report(self, claim: str, confidence: float, action: str, evidence: list | None = None) -> dict:
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {list(ACTIONS)}")
        ids = [str(e) for e in (evidence or [])]
        out = {"done": True, "claim": claim, "confidence": confidence, "action": action,
               "evidence": [e for e in ids if e in self.evidence]}
        unknown = [e for e in ids if e not in self.evidence]
        if unknown:
            out["evidence_unknown"] = unknown
        return out
