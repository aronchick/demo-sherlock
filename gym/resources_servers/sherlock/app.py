"""Native NeMo Gym resources server and verifier for Sherlock camera investigations."""
from __future__ import annotations

import json
import time
from typing import Any

from fastapi import Body, FastAPI, Request
from pydantic import PrivateAttr

from gym import raise_missing_nemo_gym
from gym.resources_servers.sherlock.scenes import (
    SCENES_BY_ID,
    SceneEpisodeState,
    SceneSpec,
    get_scene,
    score_episode,
)

try:
    from nemo_gym.base_resources_server import (
        BaseMultiRewardVerifyResponse,
        BaseResourcesServerConfig,
        BaseRunRequest,
        BaseSeedSessionRequest,
        BaseSeedSessionResponse,
        BaseVerifyRequest,
        SimpleResourcesServer,
    )
    from nemo_gym.server_utils import SESSION_ID_KEY
except ModuleNotFoundError as _ng_exc:
    raise_missing_nemo_gym(_ng_exc, cli=(__name__ == "__main__"))


class SherlockResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "sherlock"
    entrypoint: str = "app.py"


class SherlockSeedSessionRequest(BaseSeedSessionRequest):
    scene_id: str = "benign_cup_01"


class SherlockRunRequest(BaseRunRequest):
    scene_id: str = "benign_cup_01"
    category: str | None = None
    scout_event: dict[str, Any] | None = None
    expected_actions: list[str] | None = None
    require_zero_snapshots: bool = False
    require_weapon_protocol: bool = False
    require_recovery: bool = False
    require_camera_switch: str | None = None
    elapsed_ms: float | None = None


class SherlockVerifyRequest(SherlockRunRequest, BaseVerifyRequest):
    pass


class SherlockVerifyResponse(BaseMultiRewardVerifyResponse):
    scene_id: str = "benign_cup_01"
    category: str = "benign"
    task_success: float = 0.0
    tool_call_validity: float = 0.0
    steps_to_report: float = 0.0
    time_to_report_ms: float = 0.0
    action_correct: float = 0.0
    recovery_score: float = 0.0
    benign_no_snapshot: float = 0.0
    weapon_protocol_score: float = 0.0
    vlm_calls: float = 0.0
    report_summary: dict[str, Any] | None = None


_DEFAULT_SEED_BODY = Body(default_factory=SherlockSeedSessionRequest)
_DEFAULT_DICT_BODY = Body(default_factory=dict)


def _extract_scene_id_from_messages(body: SherlockRunRequest) -> str:
    if body.scene_id in SCENES_BY_ID:
        return body.scene_id
    for item in body.responses_create_params.input or []:
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else "")
        if isinstance(content, str):
            for sid, spec in SCENES_BY_ID.items():
                if f'"label": "{spec.label}"' in content and str(spec.bbox[0]) in content:
                    return sid
    return "benign_cup_01"


class SherlockResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: SherlockResourcesServerConfig
    _sessions: dict[str, SceneEpisodeState] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._sessions = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/look_at")(self.look_at)
        app.post("/move")(self.move)
        app.post("/snapshot")(self.snapshot)
        app.post("/verify_claim")(self.verify_claim)
        app.post("/list_cameras")(self.list_cameras)
        app.post("/report")(self.report)
        return app

    def _get_or_create_session(self, request: Request, scene_id: str = "benign_cup_01") -> SceneEpisodeState:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self._sessions:
            scene = get_scene(scene_id)
            self._sessions[session_id] = SceneEpisodeState(
                scene=scene,
                start_time_ms=time.perf_counter() * 1000.0,
            )
        return self._sessions[session_id]

    async def seed_session(
        self, request: Request, body: SherlockSeedSessionRequest = _DEFAULT_SEED_BODY
    ) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        scene_id = body.scene_id if body.scene_id in SCENES_BY_ID else "benign_cup_01"
        scene = get_scene(scene_id)
        self._sessions[session_id] = SceneEpisodeState(
            scene=scene,
            start_time_ms=time.perf_counter() * 1000.0,
        )
        return BaseSeedSessionResponse()

    async def look_at(
        self, request: Request, body: dict[str, Any] = _DEFAULT_DICT_BODY
    ) -> dict[str, Any]:
        state = self._get_or_create_session(request)
        return state.step_tool("look_at", body, now_ms=time.perf_counter() * 1000.0)

    async def move(
        self, request: Request, body: dict[str, Any] = _DEFAULT_DICT_BODY
    ) -> dict[str, Any]:
        state = self._get_or_create_session(request)
        return state.step_tool("move", body, now_ms=time.perf_counter() * 1000.0)

    async def snapshot(
        self, request: Request, body: dict[str, Any] = _DEFAULT_DICT_BODY
    ) -> dict[str, Any]:
        state = self._get_or_create_session(request)
        return state.step_tool("snapshot", body, now_ms=time.perf_counter() * 1000.0)

    async def verify_claim(
        self, request: Request, body: dict[str, Any] = _DEFAULT_DICT_BODY
    ) -> dict[str, Any]:
        state = self._get_or_create_session(request)
        return state.step_tool("verify", body, now_ms=time.perf_counter() * 1000.0)

    async def list_cameras(
        self, request: Request, body: dict[str, Any] = _DEFAULT_DICT_BODY
    ) -> dict[str, Any]:
        state = self._get_or_create_session(request)
        return state.step_tool("list_cameras", body, now_ms=time.perf_counter() * 1000.0)

    async def report(
        self, request: Request, body: dict[str, Any] = _DEFAULT_DICT_BODY
    ) -> dict[str, Any]:
        state = self._get_or_create_session(request)
        return state.step_tool("report", body, now_ms=time.perf_counter() * 1000.0)

    def _replay_from_response_output(
        self, scene: SceneSpec, body: SherlockVerifyRequest
    ) -> SceneEpisodeState:
        """Replay tool calls from NeMoGymResponse.output when verifying statelessly."""
        state = SceneEpisodeState(scene=scene, start_time_ms=0.0)
        for item in body.response.output or []:
            item_dict = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            if item_dict.get("type") == "function_call":
                fn_name = item_dict.get("name", "")
                raw_args_str = item_dict.get("arguments", "{}")
                try:
                    parsed_args = (
                        json.loads(raw_args_str)
                        if isinstance(raw_args_str, str)
                        else raw_args_str
                    )
                except (json.JSONDecodeError, TypeError):
                    parsed_args = raw_args_str
                state.step_tool(fn_name, parsed_args)
        return state

    async def verify(
        self, request: Request, body: Any = _DEFAULT_DICT_BODY
    ) -> SherlockVerifyResponse | dict[str, Any]:
        # Disambiguate tool call `verify(claim)` vs NeMo Gym episode verification
        if isinstance(body, dict) and "response" not in body:
            state = self._get_or_create_session(request)
            return state.step_tool("verify", body, now_ms=time.perf_counter() * 1000.0)

        verify_body = (
            body
            if isinstance(body, SherlockVerifyRequest)
            else SherlockVerifyRequest.model_validate(body)
        )
        scene_id = _extract_scene_id_from_messages(verify_body)
        scene = get_scene(scene_id)

        session_id = request.session.get(SESSION_ID_KEY) if hasattr(request, "session") else None
        state = self._sessions.get(session_id) if session_id else None
        if state is None or state.scene.scene_id != scene_id or not state.tool_trace:
            state = self._replay_from_response_output(scene, verify_body)

        fallback_ms = float(verify_body.elapsed_ms) if verify_body.elapsed_ms is not None else 45.0
        scores = score_episode(state, fallback_elapsed_ms=fallback_ms)

        reward_components = {
            "task_success": scores["task_success"],
            "tool_call_validity": scores["tool_call_validity"],
            "action_correct": scores["action_correct"],
            "recovery_score": scores["recovery_score"],
            "benign_no_snapshot": scores["benign_no_snapshot"],
            "weapon_protocol_score": scores["weapon_protocol_score"],
        }

        return SherlockVerifyResponse(
            **verify_body.model_dump(exclude={"scene_id", "category"}),
            reward=scores["reward"],
            reward_components=reward_components,
            metrics={
                "steps_to_report": scores["steps_to_report"],
                "time_to_report_ms": scores["time_to_report_ms"],
                "vlm_calls": scores["vlm_calls"],
            },
            info={
                "scene_id": scene.scene_id,
                "category": scene.category,
                "tool_trace_len": len(state.tool_trace),
                "report": state.report_payload,
            },
            scene_id=scene.scene_id,
            category=scene.category,
            task_success=scores["task_success"],
            tool_call_validity=scores["tool_call_validity"],
            steps_to_report=scores["steps_to_report"],
            time_to_report_ms=scores["time_to_report_ms"],
            action_correct=scores["action_correct"],
            recovery_score=scores["recovery_score"],
            benign_no_snapshot=scores["benign_no_snapshot"],
            weapon_protocol_score=scores["weapon_protocol_score"],
            vlm_calls=scores["vlm_calls"],
            report_summary=state.report_payload,
        )


VERIFIER_FIXTURE = {
    "server": SherlockResourcesServer,
    "config": {
        "host": "127.0.0.1",
        "port": 18080,
        "name": "sherlock",
        "entrypoint": "app.py",
    },
    "verify_method": "verify",
    "verify_request": SherlockVerifyRequest,
    "cases": "tests/verifier_cases.jsonl",
}

if __name__ == "__main__":
    SherlockResourcesServer.run_webserver()
