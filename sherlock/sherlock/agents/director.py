"""Director: the small LLM that decides whether something deserves a closer look
and drives the camera via tools. This is the Track 3 story: valid arguments,
recovery from bad/failed calls, verification, graceful fallback."""
from __future__ import annotations

import json
import re
import time

from openai import OpenAI

from ..config import Endpoint, _env
from ..events import BUS
from ..tools import TOOL_SCHEMAS, ToolBox

SYSTEM = """You are the Director of a multi-camera security system.
A cheap detector just flagged something new. The event's "threat" field decides the plan:
- "benign" (cup, bottle, phone...): call report with action="ignore" right away. No look_at, no snapshot.
- "weapon": call, in this order: look_at(camera, weapon bbox) -> snapshot(camera, subject="weapon",
  question about the object) -> look_at(camera, head_bbox) -> snapshot(camera, subject="face")
  -> report(the event claim, action="log", evidence=["weapon_crop", "face_crop"]).
  Skip the two face steps if head_bbox is null.
- "unknown": look_at -> snapshot(subject="object") -> report(action="log").
Snapshot stores evidence. Independent Analyst and Verifier checks run after acquisition finishes;
do not call verify or decide their outcome. Never ask about, describe or identify a person. The face crop is retained evidence and included in the later scene/person context.
If a tool returns an error, do not repeat the identical call:
try list_cameras and use a different camera, or adjust the arguments.
Be decisive. Use at most {max_calls} tool calls. Always finish with report."""

# The weapon chain alone is six calls; leave room to recover from a fault.
WEAPON_EXTRA_CALLS = 4

_WEAPON_CLAIMS = {"scissors": "a pair of scissors"}


def weapon_claim(label: str) -> str:
    """What the Verifier must confirm before an alert: that this is the weapon, not
    just that the Analyst's description fits the crop (a cup described as a cup
    is a supported claim, and still no reason to alert)."""
    return _WEAPON_CLAIMS.get(label, f"a {label}")


# Words that name the same object as a claim does. A claim outside these groups (an LLM
# Director's free text, a benign object) matches on any shared content word instead.
_SAME_OBJECT = (frozenset({"knife", "knive", "blade", "cleaver", "dagger", "machete"}),
                frozenset({"scissor", "shear"}),
                frozenset({"bat", "club"}),
                frozenset({"gun", "pistol", "rifle", "firearm", "handgun"}))
_ANY_WEAPON = frozenset({"weapon"}).union(*_SAME_OBJECT)  # the fine-tuned Scout's catch-all class
_FILLER = frozenset({"a", "an", "the", "of", "pair", "on", "in", "with", "and", "object", "this", "it"})


def _words(text: str) -> set[str]:
    # Plain plurals fold ("bats", "scissors"); "knives" folds to "knive", listed above.
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in re.findall(r"[a-z]+", text.lower()) if w not in _FILLER}


def names_claim(description: str, claim: str) -> bool:
    """Does the Analyst's description name the object the claim is about? SmolVLM-500M
    answers 'What is this object?' with the biggest thing in a cluttered crop: 'Cake.' for
    a knife lying beside one, while the Verifier, asked about the knife itself, confirms it.
    False means the two cannot be shown side by side."""
    said, claimed = _words(description), _words(claim)
    if "weapon" in claimed:
        return bool(said & _ANY_WEAPON)
    groups = [g for g in _SAME_OBJECT if claimed & g]
    if groups:
        return any(said & g for g in groups)
    return not claimed or bool(said & claimed)


class Director:
    def __init__(self, ep: Endpoint, tools: ToolBox, max_tool_calls: int = 6, thinking: bool | None = None):
        self.client = OpenAI(base_url=ep.base_url, api_key=ep.api_key)
        self.model = ep.model
        self.tools = tools
        self.max_calls = max_tool_calls
        if thinking is None:
            thinking = _env("DIRECTOR_THINKING", "0") == "1"
        # Nemotron 3.5 Lightning: thinking on by default; off keeps the tool loop fast.
        # vLLM and llama.cpp read the chat-template kwarg. Ollama ignores it and only turns
        # thinking off for reasoning_effort="none", which vLLM (>= 0.28) also accepts.
        # Without it, local reasoning ate the whole max_tokens and left no tool call.
        self.extra: dict = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
        if not thinking:
            self.extra["reasoning_effort"] = "none"

    def investigate(self, detection) -> dict:
        BUS.bump("investigations")
        threat = getattr(detection, "threat", "unknown")
        budget = self.max_calls + (WEAPON_EXTRA_CALLS if threat == "weapon" else 0)
        head = getattr(detection, "head", None)
        event = {"camera": detection.camera, "label": detection.label, "threat": threat,
                 "confidence": round(detection.conf, 2), "bbox": list(detection.bbox),
                 "head_bbox": list(head) if head else None,
                 "cameras": self.tools.t_list_cameras()["cameras"]}
        if threat == "weapon":
            # The claim to verify, spelled out: left to compose it, Nemotron copied the
            # prompt's "a knife" example for a Scout "scissors" (2 of 2 on 2026-10-17).
            event["claim"] = weapon_claim(detection.label)
        msgs = [{"role": "system", "content": SYSTEM.format(max_calls=budget)},
                {"role": "user", "content": json.dumps(event)}]
        calls = 0
        while calls < budget:
            t0 = time.perf_counter()
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=msgs, tools=TOOL_SCHEMAS,
                    tool_choice="auto", temperature=0.1, max_tokens=300, extra_body=self.extra)
            except Exception as e:  # noqa: BLE001 - endpoint down: degrade to RuleDirector, don't die
                BUS.emit("director", f"LLM unreachable ({e.__class__.__name__}); using rule-based fallback")
                return RuleDirector(self.tools).investigate(detection)
            m = r.choices[0].message
            BUS.emit("director", m.content or f"-> {[c.function.name for c in (m.tool_calls or [])]}",
                     (time.perf_counter() - t0) * 1000)
            msgs.append(m.model_dump(exclude_none=True))
            if not m.tool_calls:
                # Model answered in prose; nudge it back onto the rails once.
                msgs.append({"role": "user", "content": "Use the tools. Finish with report."})
                calls += 1
                continue
            for tc in m.tool_calls:
                calls += 1
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    BUS.bump("tool_errors")
                    result = {"error": f"malformed JSON arguments: {e}"}
                else:
                    result = self.tools.call(tc.function.name, args)
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
                if result.get("done"):
                    return result
        BUS.emit("director", "hit tool budget; reporting what we have")
        return {"done": True, "claim": "unresolved", "confidence": 0.0, "action": "log"}


class RuleDirector:
    """Deterministic stand-in: same tool interface, no LLM. Runs the demo in mock
    mode and is the fallback when the model endpoint is unreachable mid-pitch."""

    def __init__(self, tools: ToolBox):
        self.tools = tools

    def investigate(self, detection) -> dict:
        BUS.bump("investigations")
        threat = getattr(detection, "threat", "unknown")
        if threat == "benign":
            BUS.emit("director", f"{detection.label} is benign: ignore, no closer look")
            return self.tools.call("report", {"claim": f"benign {detection.label}", "confidence": 0.9,
                                              "action": "ignore", "evidence": []})
        subject = "weapon" if threat == "weapon" else "object"
        head = getattr(detection, "head", None)
        cams = [detection.camera] + [c for c in self.tools.cameras if c != detection.camera]
        for cam in cams:
            if self._failed(cam, self.tools.call("look_at", {"camera": cam, "bbox": list(detection.bbox)})):
                continue
            r = self.tools.call("snapshot", {"camera": cam, "subject": subject, "question": "What is this object?"})
            if self._failed(cam, r):
                continue
            claim = weapon_claim(detection.label)
            evidence = [r["evidence_id"]]
            # Face evidence is best effort: a failed zoom must not cost us the weapon alert.
            if threat == "weapon" and head and not self._failed(
                    cam, self.tools.call("look_at", {"camera": cam, "bbox": list(head)})):
                f = self.tools.call("snapshot", {"camera": cam, "subject": "face"})
                if not self._failed(cam, f):
                    evidence.append(f["evidence_id"])
            v = ({"supported": False, "confidence": 0.4} if self.tools.defer_analysis else
                 self.tools.call("verify", {"claim": claim}))
            supported = bool(v.get("supported"))
            action = "alert" if threat == "weapon" and supported else "log"
            return self.tools.call("report", {"claim": claim,
                                              "confidence": v.get("confidence", 0.4) if supported else 0.4,
                                              "action": action, "evidence": evidence})
        return self.tools.call("report", {"claim": f"{detection.label} flagged but all cameras failed",
                                          "confidence": 0.0, "action": "log"})

    @staticmethod
    def _failed(cam: str, result: dict) -> bool:
        # The ToolBox counts the recovery when a retried step succeeds.
        if "error" in result:
            BUS.emit("director", f"{cam} failed · recovering", fault=True, error=result["error"])
            return True
        return False
