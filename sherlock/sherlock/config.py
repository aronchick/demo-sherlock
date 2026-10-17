"""Environment-driven config. Every model sits behind an OpenAI-compatible URL so
Ollama (local) and vLLM (Brev) are a one-line swap."""
from __future__ import annotations

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
    yolo_model: str = field(default_factory=lambda: _env("YOLO_MODEL", "yolov8n.pt"))
    director: Endpoint = field(default_factory=lambda: Endpoint(
        _env("DIRECTOR_BASE_URL", "http://localhost:8001/v1"),
        _env("DIRECTOR_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")))
    analyst: Endpoint = field(default_factory=lambda: Endpoint(
        _env("ANALYST_BASE_URL", "http://localhost:8002/v1"),
        _env("ANALYST_MODEL", "HuggingFaceTB/SmolVLM-Instruct")))
    verifier: Endpoint = field(default_factory=lambda: Endpoint(
        _env("VERIFIER_BASE_URL", "http://localhost:8002/v1"),
        _env("VERIFIER_MODEL", "HuggingFaceTB/SmolVLM-Instruct")))
    cameras: list[CameraSpec] = field(default_factory=list)
    dashboard_port: int = field(default_factory=lambda: int(_env("DASHBOARD_PORT", "8000")))
    dwell_seconds: float = field(default_factory=lambda: float(_env("DWELL_SECONDS", "0.6")))
    max_tool_calls: int = field(default_factory=lambda: int(_env("MAX_TOOL_CALLS", "6")))
    # Nemotron 3.5 Lightning defaults to thinking mode; off = lower latency in a tool loop.
    director_thinking: bool = field(default_factory=lambda: _env("DIRECTOR_THINKING", "0") == "1")

    def __post_init__(self) -> None:
        if not self.cameras:
            for item in _env("CAMERAS", "hero:fake:0").split(","):
                name, backend, source = item.strip().split(":")
                self.cameras.append(CameraSpec(name, backend, source))


CONFIG = Config()
