"""Analyst and Verifier: small VLMs that only ever see zoomed crops.
Both talk to any OpenAI-compatible vision endpoint (Ollama /v1, vLLM)."""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import time

import cv2
import numpy as np
from openai import APIStatusError, OpenAI

from ..config import Endpoint
from ..events import BUS

log = logging.getLogger(__name__)


def _b64(img: np.ndarray, max_side: int = 768) -> str:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def _vision_request(client: OpenAI, model: str, img: np.ndarray, prompt: str, max_tokens: int = 200,
                    **extra):
    return client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=0.1,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(img)}"}},
            {"type": "text", "text": prompt}]}], **extra)


def _vision_call(client: OpenAI, model: str, img: np.ndarray, prompt: str, max_tokens: int = 200) -> str:
    r = _vision_request(client, model, img, prompt, max_tokens)
    return (r.choices[0].message.content or "").strip()


class Analyst:
    PROMPT = ("You are looking at a zoomed-in security camera crop of an object. {question} "
              "Answer in two short sentences: what the object is, and any text or brand you can read. "
              "Describe only the object. Never identify, name or describe any person. "
              "If you cannot tell, say so plainly.")

    def __init__(self, ep: Endpoint):
        self.client = OpenAI(base_url=ep.base_url, api_key=ep.api_key)
        self.model = ep.model

    def describe(self, img: np.ndarray, question: str) -> dict:
        t0 = time.perf_counter()
        text = _vision_call(self.client, self.model, img, self.PROMPT.format(question=question))
        BUS.emit("analyst", text, (time.perf_counter() - t0) * 1000)
        return {"description": text}


# Log-odds margin, log P(true) - log P(false) at the JSON verdict token, that a claim must exceed
# to count as supported. Fitted per model on the open-COCO bake-off (true vs false claims on 50
# Director-style crops). Other models decide by argmax (0.0). VERIFIER_MARGIN overrides.
MARGIN_DEFAULTS = {"HuggingFaceTB/SmolVLM-500M-Instruct": 3.31,
                   "HuggingFaceTB/SmolVLM-Instruct": 0.56}

# "supported" comes first so the verdict is the first value the model commits to.
VERDICT_SCHEMA = {"type": "object",
                  "properties": {"supported": {"type": "boolean"},
                                 "confidence": {"type": "number"},
                                 "reason": {"type": "string"}},
                  "required": ["supported", "confidence", "reason"],
                  "additionalProperties": False}
_FITTED = {k.lower(): v for k, v in MARGIN_DEFAULTS.items()}
TOP_LOGPROBS = 10
# How OpenAI-compatible servers refuse a request field they do not implement.
_REFUSED = (400, 422, 501)
_JSON_VERDICT = re.compile(r'"supported"\s*:\s*"?(true|false)\b', re.IGNORECASE)


def _bool(v) -> bool | None:
    """A JSON verdict as a bool; the string "false" must not read as truthy."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


def _lenient(text: str) -> bool | None:
    """Verdict from free text. Small VLMs answer this prompt with 'True.' or 'Supports.'
    rather than JSON; None means the reply holds no verdict at all (e.g. '0-1')."""
    if m := _JSON_VERDICT.search(text):
        return m.group(1).lower() == "true"
    t = text.strip().lower()
    if re.match(r"\W*(answer:\s*)?(true|yes|supports?|supported)\b", t):
        return True
    if re.match(r"\W*(answer:\s*)?(false|no|unsupported|not supported)\b", t) or \
            re.search(r"\b(does not|doesn't) show\b", t):
        return False
    return None


def _verdict_margin(tokens) -> float | None:
    """log P(true) - log P(false) where the model fills `"supported":`. Spellings such as
    ' true' and 'true' are pooled. A side missing from top_logprobs is bounded by the
    smallest listed logprob. None when the reply has no such token or no alternatives."""
    seen = ""
    for t in tokens:
        if t.token.strip().lower() in ("true", "false") and re.search(r'"supported"\s*:\s*$', seen):
            alts = list(t.top_logprobs or [])
            if len(alts) < 2:
                return None
            pooled: dict[str, float] = {}
            for a in alts:
                word = a.token.strip().lower()
                if word in ("true", "false"):
                    pooled[word] = float(np.logaddexp(pooled.get(word, -np.inf), a.logprob))
            floor = min(a.logprob for a in alts)
            return pooled.get("true", floor) - pooled.get("false", floor)
        seen += t.token
    return None


def margin_threshold(model: str) -> float:
    if env := os.environ.get("VERIFIER_MARGIN"):
        return float(env)
    return _FITTED.get(model.lower(), 0.0)


class Verifier:
    """Checks the Director's weapon claim against the weapon crop only.

    Small VLMs ignore the prompt's JSON instruction ('True.', 'Supports.'), and they lean
    towards yes, so the verdict is constrained to the JSON schema and decided from the
    logprobs of its boolean. An endpoint that refuses a feature is retried without it, one
    feature at a time, and every verdict records the mode that decided it:
      margin       schema + logprobs: log-odds at the verdict token > threshold
      constrained  schema, no logprobs: the boolean the model generated (argmax)
      text         no schema: JSON in the reply if any, else a leading yes/no word"""

    PROMPT = ("Claim about the object in this image: \"{claim}\". Does the image actually show that object? "
              "Judge the object only; do not identify or describe any person. "
              "Reply with JSON only: {{\"supported\": true|false, \"confidence\": 0-1, \"reason\": \"...\"}}")

    def __init__(self, ep: Endpoint):
        self.client = OpenAI(base_url=ep.base_url, api_key=ep.api_key)
        self.model = ep.model
        self.threshold = margin_threshold(ep.model)
        self.structured = self.logprobs = True  # dropped for good once the endpoint refuses them
        if "VERIFIER_MARGIN" not in os.environ and ep.model.lower() not in _FITTED:
            log.warning("verifier: no fitted margin for %r; deciding by argmax (0.0). "
                        "Set VERIFIER_MARGIN to tune it.", ep.model)

    def _ask(self, img: np.ndarray, claim: str):
        while True:
            extra: dict = {}
            if self.structured:
                extra["response_format"] = {"type": "json_schema",
                                            "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA}}
            if self.logprobs:
                extra.update(logprobs=True, top_logprobs=TOP_LOGPROBS)
            try:
                return _vision_request(self.client, self.model, img, self.PROMPT.format(claim=claim), 120,
                                       **extra)
            except APIStatusError as e:
                if e.status_code not in _REFUSED or not (self.structured or self.logprobs):
                    raise
                msg = str(e).lower()
                blames_schema = any(k in msg for k in ("response_format", "json_schema", "structured", "grammar"))
                if self.structured and (blames_schema or not self.logprobs):
                    self.structured, dropped = False, "structured outputs"
                else:
                    self.logprobs, dropped = False, "logprobs"
                log.warning("verifier: %s refused %s (HTTP %s); retrying without", self.model, dropped,
                            e.status_code)

    def check(self, img: np.ndarray, claim: str) -> dict:
        t0 = time.perf_counter()
        BUS.bump("vlm_calls")
        choice = self._ask(img, claim).choices[0]
        raw = (choice.message.content or "").strip()
        tokens = choice.logprobs.content if self.logprobs and choice.logprobs else None
        try:
            margin = _verdict_margin(tokens) if tokens else None
        except (AttributeError, TypeError, ValueError):  # a server's non-standard logprobs shape
            margin = None
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            data = {}
        data = data if isinstance(data, dict) else {}
        generated = _bool(data.get("supported"))
        if generated is None and (jm := _JSON_VERDICT.search(raw)):
            generated = jm.group(1).lower() == "true"  # schema output cut off by max_tokens
        try:
            confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        if margin is not None:
            mode, supported = "margin", margin > self.threshold
            confidence = 1 / (1 + math.exp(-margin))  # P(true) among {true, false}
        elif self.structured and generated is not None:
            mode, supported = "constrained", generated
        else:
            mode, supported = "text", bool(_lenient(raw))
        out = {"supported": supported, "confidence": round(confidence, 3),
               "reason": str(data.get("reason") or raw[:200]), "mode": mode,
               "margin": None if margin is None else round(margin, 3)}
        BUS.bump("confirmed" if supported else "rejected")
        how = f"margin {margin:+.2f} vs {self.threshold:.2f}" if margin is not None else mode
        BUS.emit("verifier", f"{'CONFIRMED' if supported else 'REJECTED'} ({how}): {out['reason']}",
                 (time.perf_counter() - t0) * 1000, **out)
        return out


# --- mocks: let the whole loop run with no model server ------------------------
# Sleeps mimic small-VLM latency so the dashboard chain is watchable; MOCK_LATENCY scales them.
def _mock_wait(seconds: float) -> float:
    s = seconds * float(os.environ.get("MOCK_LATENCY", "1"))
    time.sleep(s)
    return s * 1000


def _mock_classify(img: np.ndarray) -> str | None:
    """Crude pixel 'vision' for the synthetic scene: a near-white, unsaturated blade
    means knife; the legacy blue ACME box means cola."""
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    steel = (b > 200) & (g > 200) & (r > 200) & (np.abs(b - r) < 25)
    if steel.mean() > 0.008:
        return "knife"
    if ((img[..., 0] > 150) & (img[..., 2] < 80)).mean() > 0.05:
        return "cola"
    return None


_MOCK_TEXT = {"knife": "A kitchen knife with a bright steel blade and a dark handle. No readable text.",
              "cola": "A blue can or box with white lettering reading ACME COLA."}
_CLAIM_WORDS = {"knife": ("knife", "blade", "weapon"), "cola": ("cola", "can")}


class MockAnalyst:
    def describe(self, img: np.ndarray, question: str) -> dict:
        ms = _mock_wait(0.7)
        kind = _mock_classify(img)
        text = _MOCK_TEXT[kind] if kind else "Hard to tell; a dark object with no readable text."
        BUS.emit("analyst", text, ms)
        return {"description": text}


class MockVerifier:
    def check(self, img: np.ndarray, claim: str) -> dict:
        ms = _mock_wait(0.45)
        BUS.bump("vlm_calls")
        kind = _mock_classify(img)
        ok = kind is not None and any(w in claim.lower() for w in _CLAIM_WORDS[kind])
        out = {"supported": ok, "confidence": 0.9 if ok else 0.3,
               "reason": f"{kind} clearly visible in crop" if ok else "claim not visible in crop"}
        BUS.bump("confirmed" if ok else "rejected")
        BUS.emit("verifier", f"{'CONFIRMED' if ok else 'REJECTED'}: {out['reason']}", ms, **out)
        return out
