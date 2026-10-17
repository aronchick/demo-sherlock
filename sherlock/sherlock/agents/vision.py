"""Analyst and Verifier: small VLMs that only ever see zoomed crops.
Both talk to any OpenAI-compatible vision endpoint (Ollama /v1, vLLM)."""
from __future__ import annotations

import base64
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


# The Verifier asks a yes/no question. Small VLMs ignore a JSON-verdict instruction ('True.',
# 'Supports.') and lean towards yes under it, but answer a plain question well: on the open-COCO
# bake-off (50 Director-style crops, a true and a false claim each) SmolVLM-500M scored 94/100
# by argmax this way, against 74/100 for a schema-constrained JSON verdict with a tuned threshold.
TOP_LOGPROBS = 10
# How OpenAI-compatible servers refuse a request field they do not implement.
_REFUSED = (400, 422, 501)
_LEADING = re.compile(r"\W*(?:answer:\s*)?(yes|no|true|false)\b")


def _word(token: str) -> str:
    return re.sub(r"\W", "", token).lower()


def _leading_verdict(text: str) -> bool | None:
    """The reply's first word as a verdict ('Yes.', 'No, it is a cup.', 'Answer: yes');
    None when the reply does not open with one."""
    if m := _LEADING.match(text.strip().lower()):
        return m.group(1) in ("yes", "true")
    return None


def _verdict_margin(tokens) -> float | None:
    """log P(yes) - log P(no) at the reply's first word, pooling spellings (' Yes', 'yes',
    'YES') over top_logprobs and the sampled token. A side missing from the list is bounded
    by the smallest listed logprob. None when the reply does not open with yes or no, or the
    server lists no alternatives."""
    for t in tokens:
        word = _word(t.token)
        if not word:
            continue  # leading whitespace or punctuation
        if word not in ("yes", "no"):
            return None
        alts = {a.token: a.logprob for a in (t.top_logprobs or [])}
        if len(alts) < 2:
            return None
        alts.setdefault(t.token, t.logprob)
        pooled: dict[str, float] = {}
        for tok, lp in alts.items():
            if (w := _word(tok)) in ("yes", "no"):
                pooled[w] = float(np.logaddexp(pooled.get(w, -np.inf), lp))
        floor = min(alts.values())
        return pooled.get("yes", floor) - pooled.get("no", floor)
    return None


def margin_threshold() -> float:
    """VERIFIER_MARGIN, else 0.0 (supported when yes is the likelier answer). Fitted on the
    bake-off: SmolVLM-500M -2.75, SmolVLM-Instruct -1.88; re-tune on the room photos."""
    return float(os.environ.get("VERIFIER_MARGIN") or 0.0)


class Verifier:
    """Checks the Director's weapon claim against the weapon crop only, as a yes/no question.

    With logprobs (vLLM, llama.cpp) the verdict is the margin log P(yes) - log P(no) at the
    reply's first word against VERIFIER_MARGIN. An endpoint that refuses logprobs is retried
    without them, once and for good; without logprobs the verdict is the reply's first word.
    A reply that is neither yes nor no is not support. Every verdict records its mode:
      margin  logprobs at the first word, margin > threshold
      text    the reply's first word"""

    PROMPT = ("Is the object in this image {claim}? Judge the object only; do not identify or "
              "describe any person. Answer yes or no.")

    def __init__(self, ep: Endpoint):
        self.client = OpenAI(base_url=ep.base_url, api_key=ep.api_key)
        self.model = ep.model
        self.threshold = margin_threshold()
        self.logprobs = True  # dropped for good once the endpoint refuses them

    def _ask(self, img: np.ndarray, claim: str):
        while True:
            extra = {"logprobs": True, "top_logprobs": TOP_LOGPROBS} if self.logprobs else {}
            try:
                return _vision_request(self.client, self.model, img, self.PROMPT.format(claim=claim), 16,
                                       **extra)
            except APIStatusError as e:
                if e.status_code not in _REFUSED or not self.logprobs:
                    raise
                self.logprobs = False
                log.warning("verifier: %s refused logprobs (HTTP %s); reading the reply's first word",
                            self.model, e.status_code)

    def check(self, img: np.ndarray, claim: str) -> dict:
        t0 = time.perf_counter()
        BUS.bump("vlm_calls")
        choice = self._ask(img, claim).choices[0]
        raw = (choice.message.content or "").strip()
        try:
            tokens = choice.logprobs.content if self.logprobs and choice.logprobs else None
            margin = _verdict_margin(tokens) if tokens else None
        except (AttributeError, TypeError, ValueError):  # a server's non-standard logprobs shape
            margin = None
        if margin is not None and math.isnan(margin):
            margin = None
        if margin is not None:
            margin = max(-100.0, min(100.0, margin))  # finite, so events stay valid JSON
            mode, supported = "margin", margin > self.threshold
            confidence = round(0.5 * (1 + math.tanh(margin / 2)), 3)  # P(yes) among {yes, no}
        else:
            mode, supported, confidence = "text", _leading_verdict(raw) is True, None
        out = {"supported": supported, "confidence": confidence, "reason": raw[:200] or "(empty reply)",
               "mode": mode, "margin": None if margin is None else round(margin, 3),
               "determinate": margin is not None or _leading_verdict(raw) is not None}
        BUS.bump("confirmed" if supported else "rejected")
        how = f"margin {margin:+.2f} vs {self.threshold:+.2f}" if margin is not None else "first word"
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
