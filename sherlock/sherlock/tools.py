"""The Director's hands. Every tool returns a dict; errors come back as
{"error": ...} rather than raising, so the model gets to decide what to do."""
from __future__ import annotations

import time

import numpy as np

from .camera import Camera, CameraError, FakePTZCamera
from .events import BUS

SUBJECTS = ("weapon", "object", "face")
ACTIONS = ("log", "alert", "ignore")
_DEVICE_TOOLS = ("look_at", "move", "snapshot")

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
                       "object sends the crop to the Analyst and returns its description. subject=face only "
                       "stores the crop: it is never sent to any model. Never snapshot a benign object.",
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


class ToolBox:
    def __init__(self, cameras: dict[str, Camera], analyst, snapshot_dir: str = "snapshots", verifier=None):
        self.cameras = cameras
        self.analyst = analyst
        self.verifier = verifier
        self.snapshot_dir = snapshot_dir
        self.last_crop: np.ndarray | None = None  # latest crop of any incident, for the live view
        self.injected_failures = 0  # demo switch: next N device calls fail
        self.injected_corruptions = 0  # demo switch: next N device calls get garbage arguments
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
                         "bbox": getattr(detection, "bbox", None), "head": getattr(detection, "head", None)}
        self.evidence: dict[str, np.ndarray] = {}
        self.evidence_camera: dict[str, str] = {}  # which camera captured each crop (differs after recovery)
        self.descriptions: dict[str, str] = {}  # Analyst text per crop it saw
        self.aim: dict[str, tuple] = {}  # camera -> box its last good look_at targeted; absent = unknown
        self.verdict: dict | None = None
        self._verified: tuple | None = None  # (claim, crop id, crop version) the verdict is about
        self._versions: dict[str, int] = {}
        self._shots = 0
        self.steps = 0

    def call(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        self.steps += 1
        tag = {"step": self.steps, "tool": name, "incident": self.incident["id"]}
        if name in _DEVICE_TOOLS and self.injected_failures > 0:
            self.injected_failures -= 1
            if name != "snapshot":
                self.aim.pop(str(args.get("camera")), None)  # a failed move: pose unknown
            BUS.bump("tool_errors")
            BUS.emit("tool", f"{name} FAILED (injected)", data=args, **tag)
            return {"error": f"{name}: device returned garbage (simulated fault)"}
        if name in _DEVICE_TOOLS and self.injected_corruptions > 0:
            self.injected_corruptions -= 1
            args = {k: "\x00garbage" for k in args}  # what a confused model might send
            BUS.emit("tool", f"{name} args corrupted (injected)", **tag)
        if name == "snapshot" and self.incident["threat"] == "benign":
            BUS.emit("tool", "snapshot refused: benign object, no VLM call needed", **tag)
            return {"error": "benign object: no closer look needed. Call report with action='ignore'."}
        try:
            fn = getattr(self, f"t_{name}")
        except AttributeError:
            BUS.bump("tool_errors")
            return {"error": f"no such tool {name!r}"}
        try:
            out = fn(**args)
        except (TypeError, KeyError, ValueError, CameraError) as e:
            BUS.bump("tool_errors")
            BUS.emit("tool", f"{name} error: {e}", data=args, **tag)
            return {"error": str(e)}
        shown = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
        BUS.emit("tool", f"{name}({shown})", (time.perf_counter() - t0) * 1000, result=out, **tag)
        return out

    # --- tools -------------------------------------------------------------
    def t_look_at(self, camera: str, bbox: list) -> dict:
        if len(bbox) != 4 or any(not isinstance(v, (int, float)) for v in bbox):
            raise ValueError("bbox must be [x1,y1,x2,y2] numbers")
        cam = self._cam(camera)
        self.aim.pop(camera, None)  # a move that fails halfway leaves the pose unknown
        box = tuple(int(v) for v in bbox)
        h, w = cam.wide_frame().shape[:2]
        pose = FakePTZCamera.bbox_to_pose(box, (w, h))
        cam.move(pose.pan, pose.tilt, pose.zoom)
        self.aim[camera] = box
        return {"ok": True, **cam.describe()}

    def t_move(self, camera: str, pan=None, tilt=None, zoom=None) -> dict:
        cam = self._cam(camera)
        self.aim.pop(camera, None)  # raw degrees: nothing says what is in view now
        cam.move(pan, tilt, zoom)
        return {"ok": True, **cam.describe()}

    def aimed_at_face(self, camera: str) -> bool:
        """Is this camera's view the person's head? Decided from where the camera was
        pointed, never from the subject the Director claims, so a mislabelled head
        zoom still can't reach a model. Unknown aim counts as the head."""
        head, weapon = self.incident["head"], self.incident["bbox"]
        if head is None or weapon is None:
            return False
        aim = self.aim.get(camera)
        if aim is None:
            return True
        if _iou(aim, weapon) >= 0.5:
            return False  # the weapon box itself, even when it is held up near the face
        centre = ((aim[0] + aim[2]) / 2, (aim[1] + aim[3]) / 2)
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
        self.last_crop = img
        eid = f"{subject}_crop"
        self.evidence[eid] = img
        self.evidence_camera[eid] = camera
        self._shots += 1
        self._versions[eid] = self._shots  # a new crop invalidates any verdict on the old one
        if subject == "face":
            # Evidence only. No model ever sees this crop: no recognition, no description.
            return {"evidence_id": eid, "stored": True, "analyzed": False}
        BUS.bump("vlm_calls")
        out = self.analyst.describe(img, question)
        self.descriptions[eid] = out.get("description", "")
        return {**out, "evidence_id": eid}

    def t_verify(self, claim: str) -> dict:
        return self.check_claim(claim)

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
