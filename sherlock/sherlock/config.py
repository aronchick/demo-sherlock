"""Environment-driven config. Every model sits behind an OpenAI-compatible URL so
Ollama (local) and vLLM (Brev) are a one-line swap."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class Endpoint:
    base_url: str
    model: str
    api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", "ollama"))


@dataclass
class CameraSpec:
    name: str
    backend: str
    source: str


@dataclass
class Config:
    mock: bool = field(default_factory=lambda: _env("SHERLOCK_MOCK", "0") == "1")
    scout_backend: str = field(default_factory=lambda: _env("SCOUT_BACKEND", "yolo"))
    yolo_model: str = field(default_factory=lambda: _env("YOLO_MODEL", "models/scout-r2-960.pt"))
    scout_conf: float = field(default_factory=lambda: float(_env("SCOUT_CONF", "0.35")))
    director: Endpoint = field(default_factory=lambda: Endpoint(
        _env("DIRECTOR_BASE_URL", "http://localhost:8001/v1"),
        _env("DIRECTOR_MODEL", "RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8")))
    analyst: Endpoint = field(default_factory=lambda: Endpoint(
        _env("ANALYST_BASE_URL", "http://localhost:8002/v1"),
        _env("ANALYST_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")))
    verifier: Endpoint = field(default_factory=lambda: Endpoint(
        _env("VERIFIER_BASE_URL", "http://localhost:8002/v1"),
        _env("VERIFIER_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")))
    cameras: list[CameraSpec] = field(default_factory=list)
    camera_locations: dict[str, dict[str, str]] = field(
        default_factory=lambda: json.loads(_env("CAMERA_LOCATIONS", "{}")))
    dashboard_port: int = field(default_factory=lambda: int(_env("DASHBOARD_PORT", "8000")))
    dwell_seconds: float = field(default_factory=lambda: float(_env("DWELL_SECONDS", "0.6")))
    max_tool_calls: int = field(default_factory=lambda: int(_env("MAX_TOOL_CALLS", "6")))
    # Nemotron 3.5 Lightning defaults to thinking mode; off = lower latency in a tool loop.
    director_thinking: bool = field(default_factory=lambda: _env("DIRECTOR_THINKING", "0") == "1")

    def __post_init__(self) -> None:
        if not 0 <= self.scout_conf <= 1:
            raise ValueError("SCOUT_CONF must be between 0 and 1")
        if not self.cameras:
            for item in _env("CAMERAS", "hero:fake:0").split(","):
                name, backend, source = item.strip().split(":", 2)  # source may be "name:<camera>"
                self.cameras.append(CameraSpec(name, backend, source))
        if not isinstance(self.camera_locations, dict):
            raise TypeError("CAMERA_LOCATIONS must be an object keyed by camera name")
        names = {camera.name for camera in self.cameras}
        for camera, location in self.camera_locations.items():
            if camera not in names:
                raise ValueError("CAMERA_LOCATIONS names a camera absent from CAMERAS")
            if not isinstance(location, dict) or set(location) - {"building", "floor", "area"}:
                raise ValueError("Camera location accepts only building, floor and area")
            if any(not isinstance(value, str) or not value.strip() or len(value) > 120
                   for value in location.values()):
                raise ValueError("Camera location labels must be nonempty strings of at most 120 characters")


CONFIG = Config()
