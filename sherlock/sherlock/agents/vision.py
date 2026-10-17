"""Analyst and Verifier: small VLMs that only ever see zoomed crops.
Both talk to any OpenAI-compatible vision endpoint (Ollama /v1, vLLM)."""
from __future__ import annotations

import base64
import json
import re
import time

import cv2
import numpy as np
from openai import OpenAI

from ..config import Endpoint
from ..events import BUS


def _b64(img: np.ndarray, max_side: int = 768) -> str:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def _vision_call(client: OpenAI, model: str, img: np.ndarray, prompt: str, max_tokens: int = 200) -> str:
    r = client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=0.1,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(img)}"}},
            {"type": "text", "text": prompt}]}])
    return (r.choices[0].message.content or "").strip()


class Analyst:
    PROMPT = ("You are looking at a zoomed-in camera crop. {question} "
              "Answer in two short sentences: what the object is, and any text or brand you can read. "
              "If you cannot tell, say so plainly.")

    def __init__(self, ep: Endpoint):
        self.client = OpenAI(base_url=ep.base_url, api_key=ep.api_key)
        self.model = ep.model

    def describe(self, img: np.ndarray, question: str) -> dict:
        t0 = time.perf_counter()
        text = _vision_call(self.client, self.model, img, self.PROMPT.format(question=question))
        BUS.emit("analyst", text, (time.perf_counter() - t0) * 1000)
        return {"description": text}


class Verifier:
    PROMPT = ("Claim about this image: \"{claim}\". Does the image actually support the claim? "
              "Reply with JSON only: {{\"supported\": true|false, \"confidence\": 0-1, \"reason\": \"...\"}}")

    def __init__(self, ep: Endpoint):
        self.client = OpenAI(base_url=ep.base_url, api_key=ep.api_key)
        self.model = ep.model

    def check(self, img: np.ndarray, claim: str) -> dict:
        t0 = time.perf_counter()
        BUS.bump("vlm_calls")
        raw = _vision_call(self.client, self.model, img, self.PROMPT.format(claim=claim), 120)
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            out = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            out = {}
        out = {"supported": bool(out.get("supported", False)),
               "confidence": float(out.get("confidence", 0.0)),
               "reason": str(out.get("reason", raw[:200]))}
        BUS.bump("confirmed" if out["supported"] else "rejected")
        BUS.emit("verifier", f"{'CONFIRMED' if out['supported'] else 'REJECTED'}: {out['reason']}",
                 (time.perf_counter() - t0) * 1000, **out)
        return out


# --- mocks: let the whole loop run with no model server ------------------------
class MockAnalyst:
    def describe(self, img: np.ndarray, question: str) -> dict:
        time.sleep(0.05)
        # crude 'OCR': the synthetic source paints a blue box with white text
        blue = (img[..., 0] > 150) & (img[..., 2] < 80)
        text = "A blue can or box with white lettering reading ACME COLA." if blue.mean() > 0.05 \
            else "Hard to tell; a dark object with no readable text."
        BUS.emit("analyst", text, 50)
        return {"description": text}


class MockVerifier:
    def check(self, img: np.ndarray, claim: str) -> dict:
        time.sleep(0.03)
        BUS.bump("vlm_calls")
        ok = "cola" in claim.lower() or "can" in claim.lower()
        out = {"supported": ok, "confidence": 0.9 if ok else 0.3,
               "reason": "lettering visible in crop" if ok else "claim not visible in crop"}
        BUS.bump("confirmed" if ok else "rejected")
        BUS.emit("verifier", f"{'CONFIRMED' if ok else 'REJECTED'}: {out['reason']}", 30, **out)
        return out
