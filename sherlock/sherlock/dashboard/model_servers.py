"""Which program serves each model endpoint, asked of the server itself. This is the
one source the board (/mode), the data-flow page and the topology page name a
runtime from.

The pages must not name a runtime from a port or a plan: the local fallback runs
mlx-vlm on :8002, the port vLLM uses on Brev. Each server is identified by
something only it answers:

  vLLM     GET <base_url>/models lists models owned_by "vllm"; GET /version
  Ollama   GET /api/version
  mlx-vlm  GET /openapi.json titled "MLX-VLM Inference API" (its FastAPI app)

<ROLE>_MODEL_SERVER (DIRECTOR_, ANALYST_, VERIFIER_) or MODEL_SERVER declares the
program instead of asking. An endpoint that is neither detected nor declared is
labelled by its host, never as vLLM. Mock runs are not probed at all.

Detection is lazy with a short cache rather than once at startup, so a model
server that comes up after Sherlock is named within TTL_S."""
from __future__ import annotations

import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from ..config import Config, Endpoint

TIMEOUT_S = 1.0
TTL_S = 15.0
MLX_VLM_TITLE = "MLX-VLM Inference API"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def where(base_url: str) -> str:
    """Where an endpoint lives, for the pages: "laptop" or a remote host. An SSH
    tunnel to Brev also looks local, so BREV_HOST names the box when one is in use."""
    host = urlsplit(base_url).hostname or ""
    if host not in _LOCAL_HOSTS:
        return host
    return os.environ.get("BREV_HOST", "").strip() or "laptop"


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
    """What the server at ep says it is:
    {"name": "vLLM" | "Ollama" | "mlx-vlm" | None, "version": str | None, "reachable": bool}."""
    now = time.monotonic()
    with _lock:
        hit = _cache.get(ep.base_url)
    if hit and hit[0] > now:
        return hit[1]
    info = _probe(ep)
    with _lock:
        _cache[ep.base_url] = (now + TTL_S, info)
    return info


def server(role: str, ep: Endpoint) -> dict:
    """One role's entry: detected or declared program, its host, and the label to show."""
    host = where(ep.base_url)
    declared = (os.environ.get(f"{role.upper()}_MODEL_SERVER") or os.environ.get("MODEL_SERVER") or "").strip()
    if declared:
        info = {"name": declared, "version": None, "reachable": None, "source": "declared"}
    else:
        info = {**identify(ep), "source": "detected"}
    if info["name"]:
        label = info["name"]
    else:
        label = host if info["reachable"] else f"{host} (not answering)"
    return {**info, "host": host, "label": label}


def describe(cfg: Config) -> dict | None:
    """What serves the Director, Analyst and Verifier in a live run; None in a mock run."""
    if cfg.mock:
        return None
    roles = {"director": cfg.director, "analyst": cfg.analyst, "verifier": cfg.verifier}
    out: dict = {role: server(role, ep) for role, ep in roles.items()}
    out["endpoints"] = len({ep.base_url for ep in roles.values()})
    out["vlm_shared"] = cfg.analyst.base_url == cfg.verifier.base_url and cfg.analyst.model == cfg.verifier.model
    out["local"] = all(out[role]["host"] == "laptop" for role in roles)
    return out


def clear_cache() -> None:
    with _lock:
        _cache.clear()
