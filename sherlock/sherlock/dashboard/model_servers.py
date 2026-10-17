"""Which program serves each model endpoint, asked of the server itself.

The pages must not name a runtime from a port or a plan: the local fallback runs
mlx-vlm on :8002, the port vLLM uses on Brev. Each server is identified by
something only it answers:

  vLLM     GET <base_url>/models lists models owned_by "vllm"; GET /version
  Ollama   GET /api/version
  mlx-vlm  GET /openapi.json titled "MLX-VLM Inference API" (its FastAPI app)

Anything else is reported as reachable but unidentified, never as vLLM. Mock runs
are not probed at all."""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from ..config import Config, Endpoint

TIMEOUT_S = 1.0
TTL_S = 15.0  # a model server that comes up mid-session is named within this long
MLX_VLM_TITLE = "MLX-VLM Inference API"

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def _get(url: str, api_key: str | None = None) -> tuple[bool, dict | None]:
    """(answered, JSON object). Any HTTP response counts as answered, errors included."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            raw = r.read()
    except urllib.error.HTTPError:
        return True, None
    except (OSError, http.client.HTTPException):
        return False, None
    try:
        body = json.loads(raw)
    except ValueError:
        return True, None
    return True, body if isinstance(body, dict) else None


def _version(body: dict | None) -> str | None:
    v = (body or {}).get("version")
    return v if isinstance(v, str) else None


def _probe(ep: Endpoint) -> dict:
    parts = urlsplit(ep.base_url)
    root = f"{parts.scheme}://{parts.netloc}"
    answered, models = _get(ep.base_url.rstrip("/") + "/models", ep.api_key)
    if not answered:
        return {"name": None, "version": None, "reachable": False}
    cards = (models or {}).get("data")
    owners = {m.get("owned_by") for m in cards if isinstance(m, dict)} if isinstance(cards, list) else set()
    if any(isinstance(o, str) and o.startswith("vllm") for o in owners):
        return {"name": "vLLM", "version": _version(_get(root + "/version")[1]), "reachable": True}
    ollama = _version(_get(root + "/api/version")[1])
    if ollama:
        return {"name": "Ollama", "version": ollama, "reachable": True}
    info = (_get(root + "/openapi.json")[1] or {}).get("info")
    if isinstance(info, dict) and info.get("title") == MLX_VLM_TITLE:
        return {"name": "mlx-vlm", "version": _version(info), "reachable": True}
    return {"name": None, "version": None, "reachable": True}


def identify(ep: Endpoint) -> dict:
    """{"name": "vLLM" | "Ollama" | "mlx-vlm" | None, "version": str | None, "reachable": bool}."""
    now = time.monotonic()
    with _lock:
        hit = _cache.get(ep.base_url)
    if hit and hit[0] > now:
        return hit[1]
    info = _probe(ep)
    with _lock:
        _cache[ep.base_url] = (now + TTL_S, info)
    return info


def describe(cfg: Config) -> dict | None:
    """What serves the Director and the VLM in a live run; None in a mock run."""
    if cfg.mock:
        return None
    return {
        "director": identify(cfg.director),
        "analyst": identify(cfg.analyst),
        "verifier": identify(cfg.verifier),
        "endpoints": len({cfg.director.base_url, cfg.analyst.base_url, cfg.verifier.base_url}),
        "vlm_shared": cfg.analyst.base_url == cfg.verifier.base_url and cfg.analyst.model == cfg.verifier.model,
    }


def clear_cache() -> None:
    with _lock:
        _cache.clear()
