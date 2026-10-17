"""Director: the small LLM that decides whether something deserves a closer look
and drives the camera via tools. This is the Track 3 story: valid arguments,
recovery from bad/failed calls, verification, graceful fallback."""
from __future__ import annotations

import json
import time

from openai import OpenAI

from ..config import Endpoint, _env
from ..events import BUS
from ..tools import TOOL_SCHEMAS, ToolBox

SYSTEM = """You are the Director of a multi-camera investigation system.
A cheap detector just flagged something new. Decide if it's worth a closer look.
If yes: call look_at with the camera and bbox, then snapshot with a precise question,
then call report. If a tool returns an error, do not repeat the identical call:
try list_cameras and use a different camera, or adjust the arguments.
Be decisive. Use at most {max_calls} tool calls. Always finish with report."""


class Director:
    def __init__(self, ep: Endpoint, tools: ToolBox, max_tool_calls: int = 6, thinking: bool | None = None):
        self.client = OpenAI(base_url=ep.base_url, api_key=ep.api_key)
        self.model = ep.model
        self.tools = tools
        self.max_calls = max_tool_calls
        if thinking is None:
            thinking = _env("DIRECTOR_THINKING", "0") == "1"
        # Nemotron 3.5 Lightning: thinking on by default; off keeps the tool loop fast.
        self.extra = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}

    def investigate(self, detection) -> dict:
        BUS.bump("investigations")
        msgs = [{"role": "system", "content": SYSTEM.format(max_calls=self.max_calls)},
                {"role": "user", "content": json.dumps({
                    "camera": detection.camera, "label": detection.label,
                    "confidence": round(detection.conf, 2), "bbox": list(detection.bbox),
                    "cameras": self.tools.t_list_cameras()["cameras"]})}]
        calls = 0
        while calls < self.max_calls:
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
                if "error" in result:
                    BUS.bump("recoveries")  # model is about to get a chance to recover
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
        cams = [detection.camera] + [c for c in self.tools.cameras if c != detection.camera]
        for cam in cams:
            r = self.tools.call("look_at", {"camera": cam, "bbox": list(detection.bbox)})
            if "error" in r:
                BUS.bump("recoveries")
                BUS.emit("director", f"{cam} failed, trying next camera")
                continue
            r = self.tools.call("snapshot", {"camera": cam, "question": "What is this object? Read any text."})
            if "error" in r:
                BUS.bump("recoveries")
                continue
            desc = r["description"]
            weak = "hard to tell" in desc.lower() or "cannot" in desc.lower()
            return self.tools.call("report", {"claim": desc.split(".")[0],
                                              "confidence": 0.4 if weak else 0.85,
                                              "action": "log" if weak else "alert"})
        return self.tools.call("report", {"claim": "all cameras failed", "confidence": 0.0, "action": "log"})
