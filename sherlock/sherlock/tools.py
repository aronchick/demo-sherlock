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
        self.incident = {"id": incident_id, "threat": getattr(detection, "threat", "unknown")}
        self.evidence: dict[str, np.ndarray] = {}
        self.verdict: dict | None = None
        self.steps = 0

    def call(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        self.steps += 1
        tag = {"step": self.steps, "tool": name, "incident": self.incident["id"]}
        if name in _DEVICE_TOOLS and self.injected_failures > 0:
            self.injected_failures -= 1
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
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
        BUS.emit("tool", f"{name}({shown})", (time.perf_counter() - t0) * 1000, result=out, **tag)
        return out

    # --- tools -------------------------------------------------------------
    def t_look_at(self, camera: str, bbox: list) -> dict:
        if len(bbox) != 4 or any(not isinstance(v, (int, float)) for v in bbox):
            raise ValueError("bbox must be [x1,y1,x2,y2] numbers")
        cam = self._cam(camera)
        h, w = cam.wide_frame().shape[:2]
        pose = FakePTZCamera.bbox_to_pose(tuple(int(v) for v in bbox), (w, h))
        cam.move(pose.pan, pose.tilt, pose.zoom)
        return {"ok": True, **cam.describe()}

    def t_move(self, camera: str, pan=None, tilt=None, zoom=None) -> dict:
        cam = self._cam(camera)
        cam.move(pan, tilt, zoom)
        return {"ok": True, **cam.describe()}

    def t_snapshot(self, camera: str, subject: str = "object", question: str = "What is this object?") -> dict:
        if subject not in SUBJECTS:
            raise ValueError(f"subject must be one of {list(SUBJECTS)}")
        cam = self._cam(camera)
        img = cam.snapshot()
        self.last_crop = img
        eid = f"{subject}_crop"
        self.evidence[eid] = img
        if subject == "face":
            # Evidence only. No model ever sees this crop: no recognition, no description.
            return {"evidence_id": eid, "stored": True, "analyzed": False}
        BUS.bump("vlm_calls")
        return {**self.analyst.describe(img, question), "evidence_id": eid}

    def t_verify(self, claim: str) -> dict:
        crop = self.evidence.get("weapon_crop", self.evidence.get("object_crop"))
        if crop is None:
            raise ValueError("nothing to verify: snapshot the weapon first")
        if self.verifier is None:
            raise ValueError("no verifier configured")
        self.verdict = self.verifier.check(crop, claim)
        return self.verdict

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
