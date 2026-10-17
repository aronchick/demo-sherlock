"""Smoke tests for the NeMo Gym ablation environment inside root pytest testpaths."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("nemo_gym")


def test_nemo_gym_usage_compat_accepts_vllm_028_and_stub_usage() -> None:
    """Verify NeMo Gym usage conversion succeeds for both stub and vLLM 0.28 usage payloads."""
    from nemo_gym.openai_utils import (
        NeMoGymChatCompletion,
        NeMoGymResponseCreateParamsNonStreaming,
        NeMoGymResponseInputTokensDetails,
        NeMoGymResponseUsage,
    )
    from nemo_gym.responses_converter import ResponsesConverter

    from gym import ensure_nemo_gym_usage_compat

    ensure_nemo_gym_usage_compat()

    # Direct construction with cached_tokens=None (as responses_converter.py does) must succeed
    details = NeMoGymResponseInputTokensDetails(cached_tokens=None)
    assert details.cached_tokens is None

    converter = ResponsesConverter(return_token_id_information=False)
    req_params = NeMoGymResponseCreateParamsNonStreaming(
        input="test",
        model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    )
    for usage_dict in (
        # vLLM 0.28 without prompt_tokens_details
        {"prompt_tokens": 180, "completion_tokens": 24, "total_tokens": 204},
        # vLLM 0.28 / stub with prompt_tokens_details and completion_tokens_details
        {
            "prompt_tokens": 180,
            "completion_tokens": 24,
            "total_tokens": 204,
            "prompt_tokens_details": {"cached_tokens": 128},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    ):
        chat_completion = NeMoGymChatCompletion.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "report",
                                        "arguments": json.dumps(
                                            {
                                                "claim": "Benign cup on desk.",
                                                "confidence": 0.95,
                                                "action": "ignore",
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": usage_dict,
            }
        )
        response = converter.chat_completion_to_response(req_params, chat_completion)
        assert isinstance(response.usage, NeMoGymResponseUsage)
        assert response.usage.input_tokens == 180
        assert response.usage.output_tokens == 24


def test_validate_agent_run_response_raises_clear_runtime_error_on_error_string() -> None:
    """When agent /run returns an error string or HTTP >= 400, fail with RuntimeError, not TypeError."""
    from gym.run_ablation import _validate_agent_run_response

    with pytest.raises(
        RuntimeError,
        match="Agent '/run' failed on scene 'benign_cup_desk' for ablation 'nemotron_lightning'",
    ):
        _validate_agent_run_response(
            verified="ValidationError: cache_write_tokens Field required",
            status_code=500,
            scene_id="benign_cup_desk",
            ablation_id="nemotron_lightning",
        )


def test_ablation_smoke_single_config_and_endpoint_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run one config end-to-end in stub mode and verify --all skips unset endpoint env vars."""
    from gym.run_ablation import (
        DEFAULT_CONFIG_ORDER,
        get_config_endpoint_info,
        run_all_ablations,
        run_single_ablation,
    )

    expected = [
        ("gym/configs/ablation_prompt_v1.yaml", "prompt_v1", "DIRECTOR_BASE_URL"),
        ("gym/configs/ablation_terse_tools.yaml", "terse_tools", "DIRECTOR_BASE_URL"),
        ("gym/configs/ablation_thinking_on.yaml", "thinking_on", "DIRECTOR_BASE_URL"),
        ("gym/configs/ablation_nemotron_lightning.yaml", "nemotron_lightning", "DIRECTOR_BASE_URL"),
        ("gym/configs/ablation_qwen_director.yaml", "qwen_director", "QWEN_DIRECTOR_BASE_URL"),
    ]
    assert [p for p, _, _ in expected] == DEFAULT_CONFIG_ORDER
    for cfg_path, exp_id, exp_env in expected:
        assert get_config_endpoint_info(cfg_path) == (exp_id, exp_env)

    async def _run() -> None:
        artifact = await run_single_ablation(
            "gym/configs/ablation_nemotron_lightning.yaml",
            use_stub=True,
            output_dir=tmp_path,
        )
        assert artifact["ablation_id"] == "nemotron_lightning"
        assert artifact["execution_mode"] == "local_stub"
        assert artifact["summary_metrics"]["num_tasks"] == 8
        assert artifact["summary_metrics"]["task_success"] == 1.0
        assert (tmp_path / "ablation_stub_nemotron_lightning.json").is_file()

        # Verify --all skips qwen_director when DIRECTOR_BASE_URL is set and QWEN_DIRECTOR_BASE_URL is unset
        monkeypatch.setenv("DIRECTOR_BASE_URL", "http://127.0.0.1:9/v1")
        monkeypatch.delenv("QWEN_DIRECTOR_BASE_URL", raising=False)
        results, skipped = await run_all_ablations(
            config_paths=["gym/configs/ablation_qwen_director.yaml"],
            output_dir=tmp_path,
        )
        out = capsys.readouterr().out
        assert results == []
        assert len(skipped) == 1
        assert skipped[0]["ablation_id"] == "qwen_director"
        assert skipped[0]["endpoint_env_var"] == "QWEN_DIRECTOR_BASE_URL"
        assert "Skipping ablation 'qwen_director'" in out

    asyncio.run(_run())
