"""Scripted Sherlock scenes, tool schema transforms, and deterministic episode simulator.

Reuses TOOL_SCHEMAS from sherlock/sherlock/tools.py (imported read-only).
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure sherlock package (sherlock/sherlock) is importable without mutating repo root layout
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHERLOCK_PKG_ROOT = _REPO_ROOT / "sherlock"
if str(_SHERLOCK_PKG_ROOT) in sys.path:
    sys.path.remove(str(_SHERLOCK_PKG_ROOT))
sys.path.insert(0, str(_SHERLOCK_PKG_ROOT))
_existing_sherlock = sys.modules.get("sherlock")
if _existing_sherlock is not None:
    _existing_file = str(getattr(_existing_sherlock, "__file__", "") or "")
    if str(_SHERLOCK_PKG_ROOT / "sherlock") not in _existing_file:
        del sys.modules["sherlock"]

from sherlock.tools import TOOL_SCHEMAS  # read-only import

TERSE_DESCRIPTIONS: dict[str, str] = {
    "look_at": "Zoom camera to bbox.",
    "move": "Set camera pan/tilt/zoom.",
    "snapshot": "Ask Analyst about frame.",
    "list_cameras": "List cameras.",
    "report": "Submit final finding.",
}


def get_tool_schemas(
    mode: str = "verbose",
    responses_api_format: bool = True,
) -> list[dict[str, Any]]:
    """Return tool schemas derived from sherlock.tools.TOOL_SCHEMAS without mutating it.

    Args:
        mode: "verbose" (exact descriptions from sherlock.tools.TOOL_SCHEMAS)
              or "terse" (minimal descriptions, parameter descriptions stripped).
        responses_api_format: If True, returns OpenAI Responses API function tool dicts
              ({"type": "function", "name": ..., "description": ..., "parameters": ..., "strict": False}).
              If False, returns Chat Completions tool dicts ({"type": "function", "function": ...}).
    """
    if mode not in ("verbose", "terse"):
        raise ValueError(f"Unsupported tool_descriptions mode: {mode!r}")

    schemas = copy.deepcopy(TOOL_SCHEMAS)
    result: list[dict[str, Any]] = []
    for entry in schemas:
        fn = copy.deepcopy(entry["function"])
        name = fn["name"]
        if mode == "terse":
            fn["description"] = TERSE_DESCRIPTIONS.get(name, name)
            params = fn.get("parameters", {})
            for prop in params.get("properties", {}).values():
                prop.pop("description", None)
        if responses_api_format:
            result.append(
                {
                    "type": "function",
                    "name": name,
                    "description": fn["description"],
                    "parameters": fn["parameters"],
                    "strict": False,
                }
            )
        else:
            result.append({"type": "function", "function": fn})
    return result


def validate_tool_args(name: str, args: Any) -> tuple[bool, str | None]:
    """Validate tool call arguments against sherlock.tools.TOOL_SCHEMAS contract.

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(args, dict):
        return False, f"{name}: arguments must be a JSON object, got {type(args).__name__}"

    if name == "look_at":
        if "camera" not in args or not isinstance(args["camera"], str) or not args["camera"].strip():
            return False, "look_at: missing or invalid 'camera' string"
        bbox = args.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox)
        ):
            return False, "bbox must be [x1,y1,x2,y2] numbers"
        return True, None

    if name == "move":
        if "camera" not in args or not isinstance(args["camera"], str) or not args["camera"].strip():
            return False, "move: missing or invalid 'camera' string"
        for key in ("pan", "tilt", "zoom"):
            val = args.get(key)
            if val is not None and (isinstance(val, bool) or not isinstance(val, (int, float))):
                return False, f"move: {key!r} must be a number"
        return True, None

    if name == "snapshot":
        if "camera" not in args or not isinstance(args["camera"], str) or not args["camera"].strip():
            return False, "snapshot: missing or invalid 'camera' string"
        if "question" not in args or not isinstance(args["question"], str) or not args["question"].strip():
            return False, "snapshot: missing or invalid 'question' string"
        return True, None

    if name == "list_cameras":
        return True, None

    if name == "report":
        claim = args.get("claim")
        confidence = args.get("confidence")
        action = args.get("action")
        if not isinstance(claim, str) or not claim.strip():
            return False, "report: 'claim' must be a non-empty string"
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return False, "report: 'confidence' must be a number in [0, 1]"
        if not (0.0 <= float(confidence) <= 1.0):
            return False, "report: 'confidence' must be between 0.0 and 1.0"
        if action not in ("log", "alert", "ignore"):
            return False, "report: 'action' must be one of ['log', 'alert', 'ignore']"
        return True, None

    return False, f"no such tool {name!r}"


def _bbox_iou(a: list[int | float], b: list[int | float]) -> float:
    x_left = max(float(a[0]), float(b[0]))
    y_top = max(float(a[1]), float(b[1]))
    x_right = min(float(a[2]), float(b[2]))
    y_bottom = min(float(a[3]), float(b[3]))
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(1.0, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    area_b = max(1.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
    return inter / (area_a + area_b - inter)


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    category: str  # "benign" | "standard" | "weapon" | "fault"
    camera: str
    label: str
    conf: float
    bbox: list[int]
    frame_size: list[int] = field(default_factory=lambda: [1280, 720])
    cameras: list[str] = field(default_factory=lambda: ["hero", "scout1"])
    head_bbox: list[int] | None = None
    covered_cameras: list[str] = field(default_factory=list)
    injected_gimbal_failures: int = 0
    injected_arg_corruptions: int = 0
    primary_description: str = ""
    head_description: str = ""
    wide_description: str = "wide room view; zoom in on target bbox for details"
    expected_actions: list[str] = field(default_factory=lambda: ["log"])
    require_zero_snapshots: bool = False
    require_weapon_protocol: bool = False
    require_recovery: bool = False
    require_camera_switch: str | None = None
    expected_keywords: list[str] = field(default_factory=list)

    def to_scout_event(self) -> dict[str, Any]:
        event: dict[str, Any] = {
            "camera": self.camera,
            "label": self.label,
            "conf": self.conf,
            "bbox": list(self.bbox),
            "frame_size": list(self.frame_size),
        }
        if self.head_bbox is not None:
            event["head_bbox"] = list(self.head_bbox)
        return event

    def to_user_message(self) -> str:
        return (
            f"Scout sighting: {json.dumps(self.to_scout_event())}\n"
            f"Available cameras: {list(self.cameras)}"
        )


SCENES: list[SceneSpec] = [
    SceneSpec(
        scene_id="benign_cup_01",
        category="benign",
        camera="hero",
        label="cup",
        conf=0.88,
        bbox=[420, 310, 490, 410],
        primary_description="white ceramic coffee mug sitting on the desk",
        expected_actions=["ignore"],
        require_zero_snapshots=True,
        expected_keywords=["cup", "mug", "benign", "coffee"],
    ),
    SceneSpec(
        scene_id="benign_bottle_02",
        category="benign",
        camera="hero",
        label="bottle",
        conf=0.85,
        bbox=[180, 290, 240, 430],
        primary_description="clear reusable water bottle next to keyboard",
        expected_actions=["ignore"],
        require_zero_snapshots=True,
        expected_keywords=["bottle", "water", "benign"],
    ),
    SceneSpec(
        scene_id="standard_suitcase_03",
        category="standard",
        camera="hero",
        label="suitcase",
        conf=0.91,
        bbox=[680, 340, 920, 620],
        primary_description="brown cardboard parcel labeled FRAGILE - LAB EQUIPMENT by the doorway",
        expected_actions=["log", "alert"],
        expected_keywords=["parcel", "fragile", "equipment", "suitcase", "cardboard"],
    ),
    SceneSpec(
        scene_id="weapon_knife_hero_04",
        category="weapon",
        camera="hero",
        label="knife",
        conf=0.94,
        bbox=[600, 340, 740, 420],
        head_bbox=[540, 120, 690, 270],
        primary_description="stainless steel kitchen knife held in right hand at waist height",
        head_description="person wearing dark hoodie and red baseball cap facing north hallway",
        expected_actions=["alert"],
        require_weapon_protocol=True,
        expected_keywords=["knife", "hoodie", "cap"],
    ),
    SceneSpec(
        scene_id="weapon_scissors_scout1_05",
        category="weapon",
        camera="scout1",
        label="scissors",
        conf=0.89,
        bbox=[310, 360, 430, 450],
        head_bbox=[280, 140, 420, 280],
        primary_description="heavy metal shears held near badge lanyard",
        head_description="person in grey fleece jacket with visitor badge #14 and glasses",
        expected_actions=["alert"],
        require_weapon_protocol=True,
        expected_keywords=["scissors", "shears", "grey", "jacket", "badge", "glasses"],
    ),
    SceneSpec(
        scene_id="fault_broken_gimbal_06",
        category="fault",
        camera="hero",
        label="backpack",
        conf=0.89,
        bbox=[500, 260, 720, 560],
        injected_gimbal_failures=1,
        primary_description="unattended black nylon backpack with laptop charger cord visible",
        expected_actions=["log", "alert"],
        require_recovery=True,
        expected_keywords=["backpack", "unattended", "black", "charger"],
    ),
    SceneSpec(
        scene_id="fault_covered_lens_07",
        category="fault",
        camera="hero",
        label="backpack",
        conf=0.90,
        bbox=[520, 250, 760, 580],
        covered_cameras=["hero"],
        primary_description="black canvas backpack left beside server rack",
        expected_actions=["log", "alert"],
        require_recovery=True,
        require_camera_switch="scout1",
        expected_keywords=["backpack", "canvas", "rack", "black"],
    ),
    SceneSpec(
        scene_id="fault_corrupted_args_08",
        category="fault",
        camera="hero",
        label="suitcase",
        conf=0.86,
        bbox=[780, 300, 1020, 610],
        injected_arg_corruptions=1,
        primary_description="hard-shell pelican equipment case with yellow calibration seal",
        expected_actions=["log", "alert"],
        require_recovery=True,
        expected_keywords=["pelican", "case", "calibration", "suitcase", "equipment"],
    ),
]

SCENES_BY_ID: dict[str, SceneSpec] = {s.scene_id: s for s in SCENES}


def get_scene(scene_id: str) -> SceneSpec:
    if scene_id not in SCENES_BY_ID:
        raise KeyError(f"Unknown scene_id {scene_id!r}; available: {list(SCENES_BY_ID)}")
    return SCENES_BY_ID[scene_id]


class SceneEpisodeState:
    """Mutable per-session simulation state for one Sherlock scene."""

    def __init__(self, scene: SceneSpec, start_time_ms: float = 0.0) -> None:
        self.scene = scene
        self.start_time_ms = start_time_ms
        self.report_time_ms: float | None = None
        self.remaining_gimbal_failures = scene.injected_gimbal_failures
        self.remaining_arg_corruptions = scene.injected_arg_corruptions
        self.covered_cameras = set(scene.covered_cameras)
        self.camera_poses: dict[str, dict[str, Any]] = {
            cam: {
                "name": cam,
                "pan": 0.0,
                "tilt": 0.0,
                "zoom": 1.0,
                "status": "covered" if cam in self.covered_cameras else "ok",
                "target_region": "wide",
                "last_bbox": None,
            }
            for cam in scene.cameras
        }
        self.tool_trace: list[dict[str, Any]] = []
        self.snapshots_taken: list[dict[str, Any]] = []
        self.report_payload: dict[str, Any] | None = None
        self.encountered_error: bool = False
        self.recovered_after_error: bool = False

    def step_tool(self, name: str, raw_args: Any, now_ms: float = 0.0) -> dict[str, Any]:
        valid, err_msg = validate_tool_args(name, raw_args)
        args = raw_args if isinstance(raw_args, dict) else {}

        # Validate camera name if present
        if valid and name in ("look_at", "move", "snapshot"):
            cam_name = args["camera"]
            if cam_name not in self.camera_poses:
                valid = False
                err_msg = f"unknown camera {cam_name!r}; known: {list(self.camera_poses)}"

        if not valid:
            self.encountered_error = True
            out = {"error": err_msg or f"invalid arguments for {name}"}
            self.tool_trace.append(
                {
                    "tool": name,
                    "args": args,
                    "valid_args": False,
                    "injected_fault": False,
                    "error": True,
                    "result": out,
                }
            )
            return out

        # Check injected gimbal fault on first camera action
        if self.remaining_gimbal_failures > 0 and name in ("look_at", "move", "snapshot"):
            self.remaining_gimbal_failures -= 1
            self.encountered_error = True
            out = {"error": f"{name}: device returned garbage (simulated fault)"}
            self.tool_trace.append(
                {
                    "tool": name,
                    "args": args,
                    "valid_args": True,
                    "injected_fault": True,
                    "error": True,
                    "result": out,
                }
            )
            return out

        # Check injected argument corruption on first look_at call
        if self.remaining_arg_corruptions > 0 and name == "look_at":
            self.remaining_arg_corruptions -= 1
            self.encountered_error = True
            out = {
                "error": (
                    "bbox must be [x1,y1,x2,y2] numbers "
                    "(simulated transport corruption on first call; retry look_at with 4 integers)"
                )
            }
            self.tool_trace.append(
                {
                    "tool": name,
                    "args": args,
                    "valid_args": True,
                    "injected_fault": True,
                    "error": True,
                    "result": out,
                }
            )
            return out

        # Execute tool against simulated cameras
        if name == "look_at":
            cam_name = args["camera"]
            if cam_name in self.covered_cameras:
                self.encountered_error = True
                out = {
                    "error": (
                        f"{cam_name}: lens is covered (dark frame); "
                        "call list_cameras and switch to an uncovered camera"
                    )
                }
                self.tool_trace.append(
                    {
                        "tool": name,
                        "args": args,
                        "valid_args": True,
                        "injected_fault": True,
                        "error": True,
                        "result": out,
                    }
                )
                return out

            bbox = [int(v) for v in args["bbox"]]
            region = "primary"
            if self.scene.head_bbox and _bbox_iou(bbox, self.scene.head_bbox) >= 0.25:
                region = "head"
            elif _bbox_iou(bbox, self.scene.bbox) >= 0.25:
                region = "primary"
            else:
                # Also detect head vs primary by vertical center when agent estimates head bbox
                cy = 0.5 * (bbox[1] + bbox[3])
                primary_cy = 0.5 * (self.scene.bbox[1] + self.scene.bbox[3])
                if self.scene.head_bbox is not None and cy < primary_cy - 60:
                    region = "head"

            w, h = self.scene.frame_size
            cx = 0.5 * (bbox[0] + bbox[2])
            cy = 0.5 * (bbox[1] + bbox[3])
            pan = round(((cx / w) - 0.5) * 90.0, 1)
            tilt = round((0.5 - (cy / h)) * 50.0, 1)
            box_frac = max((bbox[2] - bbox[0]) / w, (bbox[3] - bbox[1]) / h, 0.1)
            zoom = round(min(4.0, max(1.2, 0.65 / box_frac)), 2)

            pose = self.camera_poses[cam_name]
            pose.update(
                {
                    "pan": pan,
                    "tilt": tilt,
                    "zoom": zoom,
                    "target_region": region,
                    "last_bbox": bbox,
                }
            )
            if self.encountered_error:
                self.recovered_after_error = True
            out = {
                "ok": True,
                "name": cam_name,
                "pan": pan,
                "tilt": tilt,
                "zoom": zoom,
                "status": pose["status"],
            }

        elif name == "move":
            cam_name = args["camera"]
            if cam_name in self.covered_cameras:
                self.encountered_error = True
                out = {
                    "error": (
                        f"{cam_name}: lens is covered (dark frame); "
                        "call list_cameras and switch to an uncovered camera"
                    )
                }
                self.tool_trace.append(
                    {
                        "tool": name,
                        "args": args,
                        "valid_args": True,
                        "injected_fault": True,
                        "error": True,
                        "result": out,
                    }
                )
                return out

            pose = self.camera_poses[cam_name]
            if args.get("pan") is not None:
                pose["pan"] = float(args["pan"])
            if args.get("tilt") is not None:
                pose["tilt"] = float(args["tilt"])
            if args.get("zoom") is not None:
                pose["zoom"] = float(args["zoom"])
            pose["target_region"] = "primary"
            if self.encountered_error:
                self.recovered_after_error = True
            out = {
                "ok": True,
                "name": cam_name,
                "pan": pose["pan"],
                "tilt": pose["tilt"],
                "zoom": pose["zoom"],
                "status": pose["status"],
            }

        elif name == "snapshot":
            cam_name = args["camera"]
            if cam_name in self.covered_cameras:
                self.encountered_error = True
                out = {
                    "error": (
                        f"{cam_name}: lens is covered (dark frame); "
                        "call list_cameras and switch to an uncovered camera"
                    )
                }
                self.tool_trace.append(
                    {
                        "tool": name,
                        "args": args,
                        "valid_args": True,
                        "injected_fault": True,
                        "error": True,
                        "result": out,
                    }
                )
                return out

            pose = self.camera_poses[cam_name]
            region = pose.get("target_region", "wide")
            question_lower = args["question"].lower()
            if region == "head" or (
                self.scene.head_description
                and any(k in question_lower for k in ("face", "head", "person", "who", "wearing", "subject"))
                and any(s.get("region") == "primary" for s in self.snapshots_taken)
            ):
                desc = self.scene.head_description or self.scene.primary_description
                effective_region = "head"
            elif region in ("primary", "head") or pose.get("zoom", 1.0) > 1.1:
                desc = self.scene.primary_description
                effective_region = "primary"
            else:
                desc = self.scene.primary_description or self.scene.wide_description
                effective_region = "primary"

            snap_record = {
                "camera": cam_name,
                "question": args["question"],
                "region": effective_region,
                "description": desc,
            }
            self.snapshots_taken.append(snap_record)
            if self.encountered_error:
                self.recovered_after_error = True
            out = {
                "description": desc,
                "camera": cam_name,
                "region": effective_region,
                "vlm_ms": 115.0,
            }

        elif name == "list_cameras":
            if self.encountered_error:
                self.recovered_after_error = True
            out = {
                "cameras": [
                    {
                        "name": c["name"],
                        "pan": c["pan"],
                        "tilt": c["tilt"],
                        "zoom": c["zoom"],
                        "status": c["status"],
                    }
                    for c in self.camera_poses.values()
                ]
            }

        elif name == "report":
            self.report_payload = {
                "claim": args["claim"],
                "confidence": float(args["confidence"]),
                "action": args["action"],
            }
            if self.report_time_ms is None and now_ms > 0.0:
                self.report_time_ms = now_ms
            if self.encountered_error:
                self.recovered_after_error = True
            out = {"done": True, **self.report_payload}

        else:
            out = {"error": f"no such tool {name!r}"}

        self.tool_trace.append(
            {
                "tool": name,
                "args": args,
                "valid_args": True,
                "injected_fault": False,
                "error": False,
                "result": out,
            }
        )
        return out


def score_episode(state: SceneEpisodeState, fallback_elapsed_ms: float = 0.0) -> dict[str, Any]:
    """Deterministic verifier scoring for a completed Sherlock episode."""
    scene = state.scene
    total_calls = len(state.tool_trace)
    valid_calls = sum(1 for t in state.tool_trace if t["valid_args"])
    tool_call_validity = (valid_calls / total_calls) if total_calls > 0 else 0.0

    # Steps to report
    report_indices = [i for i, t in enumerate(state.tool_trace) if t["tool"] == "report" and t["valid_args"]]
    steps_to_report = (report_indices[0] + 1) if report_indices else total_calls

    if state.report_time_ms is not None and state.start_time_ms > 0.0:
        time_to_report_ms = max(1.0, state.report_time_ms - state.start_time_ms)
    else:
        time_to_report_ms = max(1.0, fallback_elapsed_ms)

    report = state.report_payload
    has_report = report is not None
    action_correct = 1.0 if (has_report and report["action"] in scene.expected_actions) else 0.0

    # Benign zero-snapshot check
    num_snapshots = len(state.snapshots_taken)
    if scene.require_zero_snapshots:
        benign_no_snapshot = 1.0 if num_snapshots == 0 else 0.0
    else:
        benign_no_snapshot = 1.0 if num_snapshots >= 1 else 0.0

    # Weapon protocol check: must capture both primary (weapon) and head crops
    regions_captured = {s["region"] for s in state.snapshots_taken}
    if scene.require_weapon_protocol:
        has_weapon_crop = "primary" in regions_captured
        has_head_crop = "head" in regions_captured
        if has_weapon_crop and has_head_crop and action_correct == 1.0:
            weapon_protocol_score = 1.0
        elif has_weapon_crop and action_correct == 1.0:
            weapon_protocol_score = 0.5
        else:
            weapon_protocol_score = 0.0
    else:
        weapon_protocol_score = 1.0

    # Camera switch check (for covered lens)
    camera_switch_ok = True
    if scene.require_camera_switch is not None:
        used_backup = any(
            s["camera"] == scene.require_camera_switch for s in state.snapshots_taken
        )
        camera_switch_ok = used_backup

    # Recovery score
    if scene.require_recovery or state.encountered_error:
        recovery_score = (
            1.0
            if (state.recovered_after_error and has_report and camera_switch_ok and num_snapshots >= 1)
            else 0.0
        )
    else:
        recovery_score = 1.0 if has_report else 0.0

    # Overall task_success (strict deterministic binary pass/fail for the scene)
    if scene.require_zero_snapshots:
        task_success = 1.0 if (has_report and action_correct == 1.0 and num_snapshots == 0) else 0.0
    elif scene.require_weapon_protocol:
        task_success = (
            1.0
            if (has_report and action_correct == 1.0 and weapon_protocol_score == 1.0)
            else 0.0
        )
    elif scene.require_recovery:
        task_success = (
            1.0
            if (has_report and action_correct == 1.0 and recovery_score == 1.0 and num_snapshots >= 1)
            else 0.0
        )
    else:
        task_success = (
            1.0
            if (has_report and action_correct == 1.0 and num_snapshots >= 1)
            else 0.0
        )

    # Composite continuous reward in [0, 1] for RL / NeMo Gym verify response
    reward = round(
        0.50 * task_success
        + 0.20 * action_correct
        + 0.15 * tool_call_validity
        + 0.15 * min(benign_no_snapshot, weapon_protocol_score, recovery_score),
        4,
    )

    return {
        "reward": reward,
        "task_success": float(task_success),
        "tool_call_validity": round(float(tool_call_validity), 4),
        "steps_to_report": float(steps_to_report),
        "time_to_report_ms": round(float(time_to_report_ms), 2),
        "action_correct": float(action_correct),
        "recovery_score": float(recovery_score),
        "benign_no_snapshot": float(benign_no_snapshot),
        "weapon_protocol_score": float(weapon_protocol_score),
        "vlm_calls": float(num_snapshots),
    }


def build_dataset_rows(tool_mode: str = "verbose") -> list[dict[str, Any]]:
    """Build NeMo Gym JSONL dataset rows for all scripted Sherlock scenes."""
    tools = get_tool_schemas(mode=tool_mode, responses_api_format=True)
    rows: list[dict[str, Any]] = []
    for idx, scene in enumerate(SCENES):
        rows.append(
            {
                "_ng_task_index": idx,
                "_ng_rollout_index": 0,
                "scene_id": scene.scene_id,
                "category": scene.category,
                "scout_event": scene.to_scout_event(),
                "expected_actions": list(scene.expected_actions),
                "require_zero_snapshots": scene.require_zero_snapshots,
                "require_weapon_protocol": scene.require_weapon_protocol,
                "require_recovery": scene.require_recovery,
                "require_camera_switch": scene.require_camera_switch,
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": scene.to_user_message(),
                        }
                    ],
                    "tools": tools,
                },
            }
        )
    return rows
