"""End-to-end tests for Sherlock NeMo Gym YAML ablations."""
from __future__ import annotations

import asyncio
from pathlib import Path

from omegaconf import OmegaConf

from gym.run_ablation import (
    DEFAULT_CONFIG_ORDER,
    REPO_ROOT,
    format_markdown_table,
    run_single_ablation,
)


def _load_agent_and_model_cfg(rel_path: str) -> tuple[dict, dict, dict]:
    cfg = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / rel_path), resolve=True)
    assert isinstance(cfg, dict)
    res_cfg = cfg["sherlock"]["resources_servers"]["sherlock"]
    model_cfg = cfg["policy_model"]["responses_api_models"]["vllm_model"]
    agent_cfg = cfg["sherlock_agent"]["responses_api_agents"]["sherlock_agent"]
    return res_cfg, model_cfg, agent_cfg


def test_all_ablation_configs_keep_generative_models_within_33b_and_share_fixed_env() -> None:
    base_res, _base_model, _base_agent = _load_agent_and_model_cfg("gym/configs/ablation_nemotron_lightning.yaml")

    for rel_path in DEFAULT_CONFIG_ORDER:
        res_cfg, _model_cfg, agent_cfg = _load_agent_and_model_cfg(rel_path)
        # Fixed tasks and fixed verifier across all ablations
        assert res_cfg["entrypoint"] == base_res["entrypoint"] == "app.py"
        assert res_cfg["datasets"] == base_res["datasets"]

        # Total generative parameter budget <= 33B on every config
        total_b = float(agent_cfg["director_params_b"]) + float(agent_cfg["vlm_params_b"])
        assert total_b <= 33.0, f"{rel_path} exceeds 33B budget: {total_b}B"


def test_each_ablation_swaps_exactly_one_component_vs_reference() -> None:
    _, ref_model, ref_agent = _load_agent_and_model_cfg("gym/configs/ablation_nemotron_lightning.yaml")

    def _signature(model_cfg: dict, agent_cfg: dict) -> dict[str, object]:
        return {
            "model": model_cfg["model"],
            "enable_thinking": bool((model_cfg.get("chat_template_kwargs") or {}).get("enable_thinking", False)),
            "prompt_config": agent_cfg["prompt_config"],
            "tool_descriptions": agent_cfg["tool_descriptions"],
        }

    ref_sig = _signature(ref_model, ref_agent)

    expected_swapped_key = {
        "gym/configs/ablation_qwen_director.yaml": "model",
        "gym/configs/ablation_thinking_on.yaml": "enable_thinking",
        "gym/configs/ablation_terse_tools.yaml": "tool_descriptions",
        "gym/configs/ablation_prompt_v1.yaml": "prompt_config",
    }

    for rel_path, expected_key in expected_swapped_key.items():
        _, m_cfg, a_cfg = _load_agent_and_model_cfg(rel_path)
        sig = _signature(m_cfg, a_cfg)
        differing_keys = [k for k in ref_sig if sig[k] != ref_sig[k]]
        assert differing_keys == [expected_key], (
            f"{rel_path} should swap only {expected_key!r}, but differed on {differing_keys}"
        )


def test_end_to_end_ablation_pipeline_with_local_stub(tmp_path: Path) -> None:
    async def _run() -> None:
        r_v1 = await run_single_ablation(
            "gym/configs/ablation_prompt_v1.yaml",
            use_stub=True,
            output_dir=tmp_path,
        )
        r_terse = await run_single_ablation(
            "gym/configs/ablation_terse_tools.yaml",
            use_stub=True,
            output_dir=tmp_path,
        )
        r_ref = await run_single_ablation(
            "gym/configs/ablation_nemotron_lightning.yaml",
            use_stub=True,
            output_dir=tmp_path,
        )

        # Verify stub output files written with *_stub_*.json naming
        assert (tmp_path / "ablation_stub_prompt_v1.json").is_file()
        assert (tmp_path / "ablation_stub_terse_tools.json").is_file()
        assert (tmp_path / "ablation_stub_nemotron_lightning.json").is_file()

        # Prompt v2 improves task_success over prompt v1 (benign zero-snapshot + 2-crop weapon protocol)
        assert r_ref["summary_metrics"]["task_success"] > r_v1["summary_metrics"]["task_success"]
        assert r_ref["summary_metrics"]["benign_no_snapshot"] == 1.0
        assert r_v1["summary_metrics"]["benign_no_snapshot"] < 1.0

        # Verbose tool schemas improve tool_call_validity and steps_to_report over terse tool schemas
        assert r_ref["summary_metrics"]["tool_call_validity"] > r_terse["summary_metrics"]["tool_call_validity"]
        assert r_ref["summary_metrics"]["steps_to_report"] < r_terse["summary_metrics"]["steps_to_report"]

        # NeMo Gym native aggregate_metrics populated
        assert "key_metrics" in r_ref["nemo_gym_aggregate_metrics"]
        assert r_ref["nemo_gym_aggregate_metrics"]["key_metrics"]["mean/task_success"] == 1.0

        table = format_markdown_table([r_v1, r_terse, r_ref])
        assert "nemotron_lightning" in table
        assert "Tool-Call Validity" in table

    asyncio.run(_run())
