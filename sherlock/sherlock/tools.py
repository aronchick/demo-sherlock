"""The Director's hands. Every tool returns a dict; errors come back as
{"error": ...} rather than raising, so the model gets to decide what to do."""
from __future__ import annotations

import time

import numpy as np

from .camera import Camera, CameraError, FakePTZCamera
from .events import BUS

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "look_at",
        "description": "Point a camera at a box seen in its wide view and zoom so the box fills the frame.",
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
        "description": "Capture what the camera sees right now and send it to the Analyst. Returns the Analyst's description.",
        "parameters": {"type": "object", "properties": {
            "camera": {"type": "string"},
            "question": {"type": "string", "description": "What to ask the Analyst about the image"}},
            "required": ["camera", "question"]}}},
    {"type": "function", "function": {
        "name": "list_cameras",
        "description": "List cameras and their current pose/status. Use when a camera fails.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "report",
        "description": "Finish the investigation with a conclusion. Call this exactly once at the end.",
        "parameters": {"type": "object", "properties": {
            "claim": {"type": "string", "description": "One sentence: what the object is"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "action": {"type": "string", "enum": ["log", "alert", "ignore"]}},
            "required": ["claim", "confidence", "action"]}}},
]


class ToolBox:
    def __init__(self, cameras: dict[str, Camera], analyst, snapshot_dir: str = "snapshots"):
        self.cameras = cameras
        self.analyst = analyst
        self.snapshot_dir = snapshot_dir
        self.last_crop: np.ndarray | None = None
        self.injected_failures = 0  # demo switch: next N tool calls fail

    def _cam(self, name: str) -> Camera:
        if name not in self.cameras:
            raise KeyError(f"unknown camera {name!r}; known: {list(self.cameras)}")
        return self.cameras[name]

    def inject_failure(self, n: int = 1) -> None:
        self.injected_failures += n

    def call(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        if self.injected_failures > 0 and name in ("look_at", "move", "snapshot"):
            self.injected_failures -= 1
            BUS.bump("tool_errors")
            BUS.emit("tool", f"{name} FAILED (injected)", data=args)
            return {"error": f"{name}: device returned garbage (simulated fault)"}
        try:
            fn = getattr(self, f"t_{name}")
        except AttributeError:
            BUS.bump("tool_errors")
            return {"error": f"no such tool {name!r}"}
        try:
            out = fn(**args)
        except (TypeError, KeyError, ValueError, CameraError) as e:
            BUS.bump("tool_errors")
            BUS.emit("tool", f"{name} error: {e}", data=args)
            return {"error": str(e)}
        BUS.emit("tool", f"{name}({args})", (time.perf_counter() - t0) * 1000, result=out)
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

    def t_snapshot(self, camera: str, question: str) -> dict:
        cam = self._cam(camera)
        img = cam.snapshot()
        self.last_crop = img
        BUS.bump("vlm_calls")
        return self.analyst.describe(img, question)

    def t_list_cameras(self) -> dict:
        return {"cameras": [c.describe() for c in self.cameras.values()]}

    def t_report(self, claim: str, confidence: float, action: str) -> dict:
        return {"done": True, "claim": claim, "confidence": confidence, "action": action}
