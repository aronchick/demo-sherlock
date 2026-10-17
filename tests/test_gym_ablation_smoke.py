"""Smoke tests for the NeMo Gym ablation environment inside root pytest testpaths."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from gym import (
    MIN_GYM_PYTHON,
    ensure_nemo_gym_usage_compat,
    format_missing_nemo_gym_error,
    raise_missing_nemo_gym,
)


def test_missing_nemo_gym_error_message_and_python_version_pin() -> None:
    """Verify .python-version pins >= 3.13.14 and missing nemo_gym gives a clear message."""
    repo_root = Path(__file__).resolve().parents[1]
    py_ver_str = (repo_root / ".python-version").read_text().strip()
    parts = [int(p) for p in py_ver_str.split(".")]
    assert len(parts) == 3, f".python-version must specify major.minor.patch, got {py_ver_str!r}"
    assert tuple(parts) >= MIN_GYM_PYTHON, (
        f".python-version {py_ver_str!r} is below NeMo Gym minimum {MIN_GYM_PYTHON}"
    )

    msg_old = format_missing_nemo_gym_error((3, 13, 5))
    assert "install the gym extra on Python >= 3.13.14" in msg_old
    assert "3.13.5" in msg_old

    msg_new = format_missing_nemo_gym_error((3, 13, 14))
    assert "install the gym extra on Python >= 3.13.14" in msg_new

    with pytest.raises(
        ModuleNotFoundError,
        match=r"install the gym extra on Python >= 3\.13\.14",
    ):
        raise_missing_nemo_gym(ModuleNotFoundError("No module named 'nemo_gym'"), cli=False)

    with pytest.raises(
        SystemExit,
        match=r"install the gym extra on Python >= 3\.13\.14",
    ):
        raise_missing_nemo_gym(ModuleNotFoundError("No module named 'nemo_gym'"), cli=True)


def test_nemo_gym_usage_compat_accepts_vllm_028_and_stub_usage() -> None:
    """Verify NeMo Gym usage conversion succeeds for both stub and vLLM 0.28 usage payloads."""
    pytest.importorskip("nemo_gym")
    from nemo_gym.openai_utils import (
        NeMoGymChatCompletion,
        NeMoGymResponseCreateParamsNonStreaming,
        NeMoGymResponseInputTokensDetails,
        NeMoGymResponseUsage,
    )
    from nemo_gym.responses_converter import ResponsesConverter

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
    pytest.importorskip("nemo_gym")
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
    pytest.importorskip("nemo_gym")
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


def test_ollama_reasoning_effort_and_director_model_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify thinking-off configs pass reasoning_effort=none, thinking_on omits it, and DIRECTOR_MODEL overrides model IDs."""
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    thinking_off_configs = [
        "gym/configs/ablation_prompt_v1.yaml",
        "gym/configs/ablation_terse_tools.yaml",
        "gym/configs/ablation_nemotron_lightning.yaml",
        "gym/configs/ablation_qwen_director.yaml",
    ]
    thinking_on_config = "gym/configs/ablation_thinking_on.yaml"

    monkeypatch.delenv("DIRECTOR_MODEL", raising=False)
    monkeypatch.delenv("QWEN_DIRECTOR_MODEL", raising=False)

    for rel_path in thinking_off_configs:
        cfg = OmegaConf.to_container(OmegaConf.load(repo_root / rel_path), resolve=True)
        assert isinstance(cfg, dict)
        model_cfg = cfg["policy_model"]["responses_api_models"]["vllm_model"]
        extra_body = model_cfg["extra_body"]
        assert model_cfg["chat_template_kwargs"]["enable_thinking"] is False
        assert extra_body["chat_template_kwargs"]["enable_thinking"] is False
        assert extra_body.get("reasoning_effort") == "none", (
            f"{rel_path} must set extra_body.reasoning_effort='none' for Ollama compatibility"
        )

    on_cfg = OmegaConf.to_container(OmegaConf.load(repo_root / thinking_on_config), resolve=True)
    assert isinstance(on_cfg, dict)
    on_model_cfg = on_cfg["policy_model"]["responses_api_models"]["vllm_model"]
    assert on_model_cfg["chat_template_kwargs"]["enable_thinking"] is True
    assert on_model_cfg["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert "reasoning_effort" not in on_model_cfg["extra_body"]
    assert "reasoning_effort" not in on_model_cfg

    # Verify DIRECTOR_MODEL and QWEN_DIRECTOR_MODEL env overrides for local Ollama fallback
    monkeypatch.setenv("DIRECTOR_MODEL", "nemotron-3-nano:30b")
    monkeypatch.setenv("QWEN_DIRECTOR_MODEL", "qwen2.5:7b")

    for rel_path in [
        "gym/configs/ablation_prompt_v1.yaml",
        "gym/configs/ablation_terse_tools.yaml",
        "gym/configs/ablation_thinking_on.yaml",
        "gym/configs/ablation_nemotron_lightning.yaml",
    ]:
        cfg = OmegaConf.to_container(OmegaConf.load(repo_root / rel_path), resolve=True)
        assert isinstance(cfg, dict)
        model_cfg = cfg["policy_model"]["responses_api_models"]["vllm_model"]
        agent_cfg = cfg["sherlock_agent"]["responses_api_agents"]["sherlock_agent"]
        assert model_cfg["model"] == "nemotron-3-nano:30b"
        assert agent_cfg["director_model_name"] == "nemotron-3-nano:30b"

    qwen_cfg = OmegaConf.to_container(
        OmegaConf.load(repo_root / "gym/configs/ablation_qwen_director.yaml"),
        resolve=True,
    )
    assert isinstance(qwen_cfg, dict)
    assert (
        qwen_cfg["policy_model"]["responses_api_models"]["vllm_model"]["model"]
        == "qwen2.5:7b"
    )
    assert (
        qwen_cfg["sherlock_agent"]["responses_api_agents"]["sherlock_agent"]["director_model_name"]
        == "qwen2.5:7b"
    )


def test_ablation_artifacts_record_ollama_runtime_and_feed_numbers() -> None:
    """Verify measured local Ollama ablation files record Ollama runtime (never vLLM) and feed /numbers."""
    from sherlock.dashboard.numbers import read_numbers

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    expected_files = [
        "ablation_prompt_v1.json",
        "ablation_terse_tools.json",
        "ablation_thinking_on.json",
        "ablation_nemotron_lightning.json",
        "ablation_summary.json",
    ]
    for fname in expected_files:
        fpath = results_dir / fname
        assert fpath.is_file(), f"Missing measured ablation result: {fpath}"
        raw = fpath.read_text()
        assert "vllm" not in raw.lower(), f"{fname} must not claim vLLM runtime"
        payload = json.loads(raw)
        assert payload["execution_mode"] == "live_endpoint"
        assert payload["runtime"] == "local_ollama"
        assert payload["model_server"]["name"] == "Ollama"
        assert payload["model_server"]["host"] == "laptop"

    nums = read_numbers(repo_root)
    assert nums["ablation"] is not None
    assert nums["ablation"]["config"] == "nemotron_lightning"
    summary = json.loads((results_dir / "ablation_summary.json").read_text())
    reference = next(row for row in summary["results"] if row["ablation_id"] == "nemotron_lightning")
    metrics = reference["summary_metrics"]
    assert nums["ablation"]["file"] == "ablation_summary.json"
    assert nums["ablation"]["tasks"] == metrics["num_tasks"]
    assert nums["ablation"]["task_success"] == metrics["task_success"]
    assert nums["ablation"]["tool_call_validity"] == metrics["tool_call_validity"]
    assert nums["ablation"]["runtime"] == "local_ollama"
    assert nums["ablation"]["model_server_label"] == "Ollama"


def test_brev_vllm_ablation_artifacts_record_vllm_runtime() -> None:
    """Verify measured Brev H100 vLLM ablation files record vLLM 0.28.0 on sherlock-h100."""
    from sherlock.dashboard.numbers import ablation_result

    repo_root = Path(__file__).resolve().parents[1]
    brev_dir = repo_root / "results" / "brev_vllm"
    expected_files = [
        "ablation_prompt_v1.json",
        "ablation_terse_tools.json",
        "ablation_thinking_on.json",
        "ablation_nemotron_lightning.json",
        "ablation_summary.json",
    ]
    for fname in expected_files:
        fpath = brev_dir / fname
        assert fpath.is_file(), f"Missing measured Brev vLLM ablation result: {fpath}"
        payload = json.loads(fpath.read_text())
        assert payload["execution_mode"] == "live_endpoint"
        assert payload["runtime"] == "brev_vllm"
        assert payload["model_server"]["name"] == "vLLM"
        assert payload["model_server"]["version"] == "0.28.0"
        assert payload["model_server"]["host"] == "sherlock-h100"
        assert payload["model_server"]["label"] == "vLLM"

    summary = json.loads((brev_dir / "ablation_summary.json").read_text())
    parsed = ablation_result(summary)
    assert parsed is not None
    assert parsed["config"] == "nemotron_lightning"
    assert parsed["runtime"] == "brev_vllm"
    assert parsed["model_server_label"] == "vLLM"
    assert parsed["tasks"] == 8
    assert parsed["task_success"] == 1.0
    assert parsed["tool_call_validity"] == 1.0



