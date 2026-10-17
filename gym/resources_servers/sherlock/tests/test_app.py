"""Unit tests for SherlockResourcesServer and deterministic verifier."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient

from gym.resources_servers.sherlock.app import (
    SherlockResourcesServer,
    SherlockResourcesServerConfig,
    SherlockSeedSessionRequest,
    SherlockVerifyRequest,
)
from gym.resources_servers.sherlock.scenes import (
    TOOL_SCHEMAS,
    get_scene,
    get_tool_schemas,
    validate_tool_args,
)


def _make_server() -> SherlockResourcesServer:
    config = SherlockResourcesServerConfig(host="127.0.0.1", port=18080, name="sherlock")
    client = MagicMock(spec=ServerClient)
    return SherlockResourcesServer(config=config, server_client=client)


def _mock_request(session_id: str = "sess-1") -> MagicMock:
    req = MagicMock()
    req.session = {SESSION_ID_KEY: session_id}
    return req


def test_tool_schemas_imported_from_sherlock_tools() -> None:
    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    assert names == ["look_at", "move", "snapshot", "list_cameras", "report"]

    verbose_tools = get_tool_schemas(mode="verbose", responses_api_format=True)
    terse_tools = get_tool_schemas(mode="terse", responses_api_format=True)
    assert len(verbose_tools) == 5
    assert len(terse_tools) == 5
    # Verify original TOOL_SCHEMAS was not mutated
    assert "[x1, y1, x2, y2]" in TOOL_SCHEMAS[0]["function"]["parameters"]["properties"]["bbox"]["description"]
    # Verify terse stripped parameter descriptions
    assert "description" not in terse_tools[0]["parameters"]["properties"]["bbox"]


def test_validate_tool_args_contract() -> None:
    ok, err = validate_tool_args("look_at", {"camera": "hero", "bbox": [10, 20, 100, 200]})
    assert ok and err is None

    ok, err = validate_tool_args("look_at", {"camera": "hero", "bbox": "10,20,100,200"})
    assert not ok and err is not None

    ok, err = validate_tool_args("report", {"claim": "cup", "confidence": 0.9, "action": "ignore"})
    assert ok and err is None

    ok, err = validate_tool_args("report", {"claim": "cup", "confidence": 1.5, "action": "ignore"})
    assert not ok

    ok, err = validate_tool_args("report", {"claim": "cup", "confidence": 0.9, "action": " panic "})
    assert not ok


def test_benign_scene_zero_snapshots_vs_unnecessary_snapshot() -> None:
    async def _run() -> None:
        server = _make_server()
        req = _mock_request("benign-ok")
        await server.seed_session(req, SherlockSeedSessionRequest(scene_id="benign_cup_01"))
        res = await server.report(req, {"claim": "Coffee mug on desk", "confidence": 0.95, "action": "ignore"})
        assert res["done"] is True

        verify_req = SherlockVerifyRequest(
            scene_id="benign_cup_01",
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="resp_1",
                created_at=0.0,
                model="stub",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            ),
        )
        verified = await server.verify(req, verify_req)
        assert verified.task_success == 1.0
        assert verified.benign_no_snapshot == 1.0
        assert verified.steps_to_report == 1.0

        # Now test when agent wastes a snapshot on benign cup
        req_bad = _mock_request("benign-bad")
        await server.seed_session(req_bad, SherlockSeedSessionRequest(scene_id="benign_cup_01"))
        await server.look_at(req_bad, {"camera": "hero", "bbox": [420, 310, 490, 410]})
        await server.snapshot(req_bad, {"camera": "hero", "question": "Is this a cup?"})
        await server.report(req_bad, {"claim": "Coffee mug", "confidence": 0.9, "action": "ignore"})
        verified_bad = await server.verify(req_bad, verify_req)
        assert verified_bad.benign_no_snapshot == 0.0
        assert verified_bad.task_success == 0.0

    asyncio.run(_run())


def test_weapon_scene_two_crop_protocol() -> None:
    async def _run() -> None:
        server = _make_server()
        scene = get_scene("weapon_knife_hero_04")
        req = _mock_request("weapon-full")
        await server.seed_session(req, SherlockSeedSessionRequest(scene_id=scene.scene_id))

        await server.look_at(req, {"camera": "hero", "bbox": scene.bbox})
        snap1 = await server.snapshot(req, {"camera": "hero", "question": "Confirm weapon"})
        assert "knife" in snap1["description"]

        assert scene.head_bbox is not None
        await server.look_at(req, {"camera": "hero", "bbox": scene.head_bbox})
        snap2 = await server.snapshot(req, {"camera": "hero", "question": "Describe face/head"})
        assert "hoodie" in snap2["description"]

        await server.report(
            req,
            {
                "claim": "Person in dark hoodie and red cap holding stainless steel knife",
                "confidence": 0.96,
                "action": "alert",
            },
        )

        verify_req = SherlockVerifyRequest(
            scene_id=scene.scene_id,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="resp_w",
                created_at=0.0,
                model="stub",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            ),
        )
        verified = await server.verify(req, verify_req)
        assert verified.weapon_protocol_score == 1.0
        assert verified.task_success == 1.0
        assert verified.steps_to_report == 5.0

    asyncio.run(_run())


def test_covered_lens_and_broken_gimbal_recovery() -> None:
    async def _run() -> None:
        server = _make_server()

        # Covered lens on hero -> must list_cameras and switch to scout1
        req_cov = _mock_request("cov-1")
        await server.seed_session(req_cov, SherlockSeedSessionRequest(scene_id="fault_covered_lens_07"))
        err_res = await server.look_at(req_cov, {"camera": "hero", "bbox": [520, 250, 760, 580]})
        assert "error" in err_res and "covered" in err_res["error"]

        cams = await server.list_cameras(req_cov, {})
        statuses = {c["name"]: c["status"] for c in cams["cameras"]}
        assert statuses == {"hero": "covered", "scout1": "ok"}

        await server.look_at(req_cov, {"camera": "scout1", "bbox": [520, 250, 760, 580]})
        snap = await server.snapshot(req_cov, {"camera": "scout1", "question": "What is the object?"})
        assert "backpack" in snap["description"]
        await server.report(req_cov, {"claim": "Black canvas backpack", "confidence": 0.9, "action": "log"})

        verify_req = SherlockVerifyRequest(
            scene_id="fault_covered_lens_07",
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="resp_cov",
                created_at=0.0,
                model="stub",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            ),
        )
        verified = await server.verify(req_cov, verify_req)
        assert verified.recovery_score == 1.0
        assert verified.task_success == 1.0
        assert verified.tool_call_validity == 1.0

    asyncio.run(_run())


def test_stateless_verify_replay_with_corrupted_tool_call() -> None:
    async def _run() -> None:
        server = _make_server()
        req = _mock_request("stateless-replay")
        output_calls = [
            NeMoGymResponseFunctionToolCall(
                type="function_call",
                call_id="c1",
                name="look_at",
                arguments=json.dumps({"camera": "hero", "bbox": "not-a-list"}),
            ),
            NeMoGymResponseFunctionToolCall(
                type="function_call",
                call_id="c2",
                name="look_at",
                arguments=json.dumps({"camera": "hero", "bbox": [680, 340, 920, 620]}),
            ),
            NeMoGymResponseFunctionToolCall(
                type="function_call",
                call_id="c3",
                name="snapshot",
                arguments=json.dumps({"camera": "hero", "question": "Read label"}),
            ),
            NeMoGymResponseFunctionToolCall(
                type="function_call",
                call_id="c4",
                name="report",
                arguments=json.dumps({"claim": "Fragile parcel", "confidence": 0.9, "action": "log"}),
            ),
        ]
        verify_req = SherlockVerifyRequest(
            scene_id="standard_suitcase_03",
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="resp_replay",
                created_at=0.0,
                model="stub",
                object="response",
                output=output_calls,
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            ),
        )
        verified = await server.verify(req, verify_req)
        assert verified.tool_call_validity == 0.75  # 3 valid out of 4 calls
        assert verified.recovery_score == 1.0
        assert verified.task_success == 1.0

    asyncio.run(_run())
