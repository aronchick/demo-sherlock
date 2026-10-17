"""Local OpenAI-compatible stub model server (/v1/chat/completions) for offline validation.

Inspects the actual OpenAI Chat Completions payload produced by NeMo Gym's VLLMModel
server (model name, system prompt content, tool schema descriptions, thinking mode,
and tool-call history) so YAML config ablations execute end-to-end through the native
NeMo Gym server stack when DIRECTOR_BASE_URL is not set.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import Body, FastAPI

BENIGN_LABELS = {"cup", "mug", "bottle", "book", "chair"}
WEAPON_LABELS = {"knife", "scissors", "gun", "weapon"}


def _parse_scout_context(messages: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    scout_event: dict[str, Any] = {
        "camera": "hero",
        "label": "suitcase",
        "conf": 0.9,
        "bbox": [400, 300, 600, 500],
        "frame_size": [1280, 720],
    }
    cameras = ["hero", "scout1"]
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            text = msg["content"]
            for line in text.splitlines():
                if line.startswith("Scout sighting:"):
                    raw_json = line.split("Scout sighting:", 1)[1].strip()
                    try:
                        scout_event = json.loads(raw_json)
                    except json.JSONDecodeError:
                        pass
                elif line.startswith("Available cameras:"):
                    raw_cams = line.split("Available cameras:", 1)[1].strip()
                    try:
                        cameras = json.loads(raw_cams.replace("'", '"'))
                    except json.JSONDecodeError:
                        pass
    return scout_event, cameras


def _extract_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract ordered pairs of assistant tool_calls and their tool results."""
    calls_by_id: dict[str, dict[str, Any]] = {}
    ordered_calls: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                call_id = tc.get("id", "")
                try:
                    parsed_args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    parsed_args = {}
                rec = {
                    "id": call_id,
                    "name": fn.get("name", ""),
                    "args": parsed_args if isinstance(parsed_args, dict) else {},
                    "result": None,
                }
                calls_by_id[call_id] = rec
                ordered_calls.append(rec)
        elif role == "tool":
            call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "{}")
            try:
                parsed_res = json.loads(content) if isinstance(content, str) else content
            except (json.JSONDecodeError, TypeError):
                parsed_res = {"raw": content}
            if call_id in calls_by_id:
                calls_by_id[call_id]["result"] = parsed_res
    return ordered_calls


def _has_verbose_tool_descriptions(tools: list[dict[str, Any]]) -> bool:
    for t in tools:
        fn = t.get("function", t)
        if fn.get("name") == "look_at":
            params = fn.get("parameters", {})
            bbox_prop = params.get("properties", {}).get("bbox", {})
            if "[x1, y1, x2, y2]" in str(bbox_prop.get("description", "")):
                return True
    return False


def _decide_next_tool_call(
    model: str,
    system_prompt: str,
    tools: list[dict[str, Any]],
    enable_thinking: bool,
    scout_event: dict[str, Any],
    cameras: list[str],
    history: list[dict[str, Any]],
) -> tuple[str, dict[str, Any], str]:
    """Return (tool_name, tool_args, optional_reasoning) based on conversation state."""
    label = str(scout_event.get("label", "object"))
    primary_cam = str(scout_event.get("camera", "hero"))
    bbox = list(scout_event.get("bbox", [400, 300, 600, 500]))
    head_bbox = scout_event.get("head_bbox")

    has_benign_rule = "Benign everyday items" in system_prompt
    has_weapon_rule = "two-crop" in system_prompt or "head_bbox" in system_prompt
    verbose_tools = _has_verbose_tool_descriptions(tools)
    is_qwen = "qwen" in model.lower()

    step_idx = len(history)
    reasoning = ""
    if enable_thinking:
        reasoning = (
            f"Analyzing Scout sighting of {label} on {primary_cam} at bbox {bbox} "
            f"(step {step_idx + 1}). Checking camera health, tool constraints, and reporting protocol."
        )

    # 1. Benign scenes
    if label in BENIGN_LABELS:
        # With system_v2 and thinking off, immediately report ignore with zero snapshots
        if has_benign_rule and not (enable_thinking and label == "bottle"):
            return (
                "report",
                {
                    "claim": f"Benign {label} spotted on {primary_cam}; no VLM inspection needed.",
                    "confidence": 0.94,
                    "action": "ignore",
                },
                reasoning,
            )
        # Without benign rule (system_v1) or when over-thinking bottle, model zooms & snapshots first
        if step_idx == 0:
            return ("look_at", {"camera": primary_cam, "bbox": bbox}, reasoning)
        if step_idx == 1:
            return (
                "snapshot",
                {"camera": primary_cam, "question": f"Confirm whether this {label} is benign."},
                reasoning,
            )
        return (
            "report",
            {
                "claim": f"Confirmed benign {label} on desk.",
                "confidence": 0.92,
                "action": "ignore",
            },
            reasoning,
        )

    # Determine active camera (switch to uncovered backup if primary failed with covered lens)
    active_cam = primary_cam
    discovered_backup: str | None = None
    for call in history:
        res = call.get("result") or {}
        if call["name"] == "list_cameras" and isinstance(res.get("cameras"), list):
            for cam_info in res["cameras"]:
                if cam_info.get("status") == "ok" and cam_info.get("name") != primary_cam:
                    discovered_backup = cam_info["name"]
                    break
    if discovered_backup:
        active_cam = discovered_backup

    # Check if the most recent tool call returned an error
    last_call = history[-1] if history else None
    last_res = (last_call.get("result") or {}) if last_call else {}
    last_err = str(last_res.get("error", "")) if isinstance(last_res, dict) else ""

    if last_err:
        if "covered" in last_err and not discovered_backup:
            # Qwen retries hero once with move before listing cameras
            if is_qwen and sum(1 for c in history if (c.get("result") or {}).get("error")) == 1:
                return ("move", {"camera": primary_cam, "pan": 0.0, "tilt": 0.0, "zoom": 2.0}, reasoning)
            return ("list_cameras", {}, reasoning)
        if "garbage" in last_err or "simulated fault" in last_err:
            # Recover by retrying look_at on active camera
            return ("look_at", {"camera": active_cam, "bbox": bbox}, reasoning)
        if "bbox must be" in last_err or "missing or invalid" in last_err:
            # Recover from corrupted or malformed arguments by sending valid schema args
            return ("look_at", {"camera": active_cam, "bbox": bbox}, reasoning)

    # Under terse tool descriptions, the model doesn't see "[x1, y1, x2, y2]" or required snapshot question
    if not verbose_tools and step_idx == 0 and label in ("scissors", "suitcase"):
        # Emit malformed bbox dict on first attempt before recovering on step 2
        return (
            "look_at",
            {"camera": active_cam, "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}},  # type: ignore[dict-item]
            reasoning,
        )

    # Count successful look_at and snapshots so far
    ok_snapshots = [
        c for c in history
        if c["name"] == "snapshot" and isinstance(c.get("result"), dict) and "description" in c["result"]
    ]
    ok_looks = [
        c for c in history
        if c["name"] in ("look_at", "move") and isinstance(c.get("result"), dict) and c["result"].get("ok")
    ]

    # 2. Weapon scenes
    if label in WEAPON_LABELS:
        if len(ok_looks) == 0:
            return ("look_at", {"camera": active_cam, "bbox": bbox}, reasoning)
        if len(ok_snapshots) == 0:
            if not verbose_tools and sum(1 for c in history if c["name"] == "snapshot") == 0:
                # Under terse tools, omit required 'question' field once
                return ("snapshot", {"camera": active_cam}, reasoning)
            return (
                "snapshot",
                {"camera": active_cam, "question": f"Verify the {label} in the subject's hand."},
                reasoning,
            )
        # Second crop (head_bbox) required by weapon protocol in system_v2
        # Note: Qwen misses the second crop on secondary camera (scout1)
        do_second_crop = has_weapon_rule and head_bbox is not None and not (is_qwen and active_cam == "scout1")
        if do_second_crop:
            if len(ok_looks) < 2:
                return ("look_at", {"camera": active_cam, "bbox": list(head_bbox)}, reasoning)
            if len(ok_snapshots) < 2:
                return (
                    "snapshot",
                    {
                        "camera": active_cam,
                        "question": "Describe the person's head, face, and clothing for identification.",
                    },
                    reasoning,
                )

        descriptions = [str(s["result"].get("description", "")) for s in ok_snapshots]
        evidence_str = "; ".join(d for d in descriptions if d)
        return (
            "report",
            {
                "claim": f"Weapon alert ({label}) on {active_cam}: {evidence_str}",
                "confidence": 0.95,
                "action": "alert",
            },
            reasoning,
        )

    # 3. Standard & fault-recovery scenes (suitcase, backpack, etc.)
    if len(ok_looks) == 0:
        return ("look_at", {"camera": active_cam, "bbox": bbox}, reasoning)
    if len(ok_snapshots) == 0:
        return (
            "snapshot",
            {"camera": active_cam, "question": f"What is this {label} and are there any markings?"},
            reasoning,
        )

    desc = str(ok_snapshots[-1]["result"].get("description", label))
    return (
        "report",
        {
            "claim": f"Inspected {label} on {active_cam}: {desc}",
            "confidence": 0.89,
            "action": "log",
        },
        reasoning,
    )


_DEFAULT_DICT_BODY = Body(default_factory=dict)


def create_stub_app() -> FastAPI:
    app = FastAPI(title="Sherlock Local OpenAI-Compatible Stub Model Server")

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                    "object": "model",
                    "owned_by": "nvidia",
                },
                {
                    "id": "Qwen/Qwen2.5-7B-Instruct",
                    "object": "model",
                    "owned_by": "qwen",
                },
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict[str, Any] = _DEFAULT_DICT_BODY) -> dict[str, Any]:
        model = str(body.get("model", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"))
        messages = body.get("messages") or []
        tools = body.get("tools") or []

        chat_template_kwargs = body.get("chat_template_kwargs") or (
            (body.get("extra_body") or {}).get("chat_template_kwargs") or {}
        )
        enable_thinking = bool(chat_template_kwargs.get("enable_thinking", False))

        # Simulate realistic inference timing differences (thinking on vs off, 30B-A3B MoE vs 7B dense)
        if enable_thinking:
            await asyncio.sleep(0.035)
        elif "qwen" in model.lower():
            await asyncio.sleep(0.014)
        else:
            await asyncio.sleep(0.008)

        system_prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = str(msg.get("content", ""))
                break

        scout_event, cameras = _parse_scout_context(messages)
        history = _extract_history(messages)

        tool_name, tool_args, reasoning = _decide_next_tool_call(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            enable_thinking=enable_thinking,
            scout_event=scout_event,
            cameras=cameras,
            history=history,
        )

        call_id = f"call_{uuid.uuid4().hex[:12]}"
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": f"<think>{reasoning}</think>" if reasoning else None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args),
                    },
                }
            ],
        }

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": assistant_msg,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 180,
                "completion_tokens": 64 if enable_thinking else 24,
                "total_tokens": 244 if enable_thinking else 204,
            },
        }

    return app
