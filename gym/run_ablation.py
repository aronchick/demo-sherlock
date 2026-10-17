"""One-command NeMo Gym ablation runner for Sherlock Director configurations.

Usage:
  # Run a single ablation config (uses DIRECTOR_BASE_URL if set, otherwise local stub):
  uv run python -m gym.run_ablation --config gym/configs/ablation_nemotron_lightning.yaml

  # Run the full 5-config ablation suite and write results/ablation_*.json:
  uv run python -m gym.run_ablation --all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import uvicorn
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    global_aiohttp_client_exit,
    is_global_aiohttp_client_setup,
    set_global_aiohttp_client,
)
from omegaconf import OmegaConf
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig

from gym.resources_servers.sherlock.app import (
    SherlockResourcesServer,
    SherlockResourcesServerConfig,
)
from gym.responses_api_agents.sherlock_agent.app import (
    SherlockAgent,
    SherlockAgentConfig,
)
from gym.stub_model_server import create_stub_app

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_ORDER = [
    "gym/configs/ablation_prompt_v1.yaml",
    "gym/configs/ablation_terse_tools.yaml",
    "gym/configs/ablation_thinking_on.yaml",
    "gym/configs/ablation_qwen_director.yaml",
    "gym/configs/ablation_nemotron_lightning.yaml",
]


def _allocate_free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            ports.append(int(s.getsockname()[1]))
            sockets.append(s)
    finally:
        for s in sockets:
            s.close()
    return ports


async def _start_uvicorn_server(app: Any, host: str, port: int) -> tuple[uvicorn.Server, asyncio.Task[None]]:
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="error",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config=config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    task = asyncio.create_task(server.serve())
    deadline = time.monotonic() + 10.0
    while not server.started:
        if task.done():
            exc = task.exception()
            raise RuntimeError(f"Server on {host}:{port} exited before starting: {exc}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for server on {host}:{port} to start")
        await asyncio.sleep(0.02)
    return server, task


async def _stop_uvicorn_servers(servers: list[tuple[uvicorn.Server, asyncio.Task[None]]]) -> None:
    for srv, _ in servers:
        srv.should_exit = True
    for _, task in servers:
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()


def _prepare_gym_fastapi_app(server: Any) -> Any:
    app = server.setup_webserver()
    server.setup_liveness(app)
    server.setup_exception_middleware(app)
    return app


async def run_single_ablation(
    config_path: str | Path,
    use_stub: bool | None = None,
    output_dir: str | Path = "results",
) -> dict[str, Any]:
    """Run one NeMo Gym YAML ablation configuration end-to-end and save JSON results."""
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        cfg_file = REPO_ROOT / config_path
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Ablation config not found: {config_path}")

    raw_cfg = OmegaConf.load(cfg_file)
    resolved_dict: dict[str, Any] = OmegaConf.to_container(raw_cfg, resolve=True)  # type: ignore[assignment]

    env_director_url = os.environ.get("DIRECTOR_BASE_URL", "").strip()
    if use_stub is None:
        effective_use_stub = not bool(env_director_url)
    else:
        effective_use_stub = use_stub

    ports = _allocate_free_ports(4)
    res_port, model_port, agent_port, stub_port = ports
    host = "127.0.0.1"

    # Inject dynamic local ports into resolved server config dictionary
    res_block = resolved_dict["sherlock"]["resources_servers"]["sherlock"]
    res_block["host"] = host
    res_block["port"] = res_port

    model_block = resolved_dict["policy_model"]["responses_api_models"]["vllm_model"]
    model_block["host"] = host
    model_block["port"] = model_port
    if effective_use_stub:
        model_block["base_url"] = f"http://{host}:{stub_port}/v1"
        model_block["api_key"] = "EMPTY"
    elif env_director_url:
        model_block["base_url"] = env_director_url

    agent_block = resolved_dict["sherlock_agent"]["responses_api_agents"]["sherlock_agent"]
    agent_block["host"] = host
    agent_block["port"] = agent_port

    global_config_dict = OmegaConf.create(resolved_dict)
    head_server_config = BaseServerConfig(host=host, port=res_port)
    server_client = ServerClient(
        head_server_config=head_server_config,
        global_config_dict=global_config_dict,
    )

    if not is_global_aiohttp_client_setup():
        set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())

    res_config = SherlockResourcesServerConfig.model_validate(res_block | {"name": "sherlock"})
    res_server = SherlockResourcesServer(config=res_config, server_client=server_client)

    model_config = VLLMModelConfig.model_validate(model_block | {"name": "policy_model"})
    model_server = VLLMModel(config=model_config, server_client=server_client)

    agent_config = SherlockAgentConfig.model_validate(agent_block | {"name": "sherlock_agent"})
    agent_server = SherlockAgent(config=agent_config, server_client=server_client)

    total_params_b = agent_config.director_params_b + agent_config.vlm_params_b
    if total_params_b > 33.0:
        raise ValueError(
            f"Ablation {agent_config.ablation_id!r} exceeds 33B generative parameter cap: "
            f"{total_params_b:.2f}B > 33.0B"
        )

    datasets = res_block.get("datasets") or []
    dataset_rel = datasets[0]["jsonl_fpath"] if datasets else "gym/resources_servers/sherlock/data/tasks.jsonl"
    dataset_path = REPO_ROOT / dataset_rel
    tasks = [
        json.loads(line)
        for line in dataset_path.read_text().splitlines()
        if line.strip()
    ]

    running_servers: list[tuple[uvicorn.Server, asyncio.Task[None]]] = []
    try:
        if effective_use_stub:
            stub_app = create_stub_app()
            running_servers.append(await _start_uvicorn_server(stub_app, host, stub_port))

        running_servers.append(await _start_uvicorn_server(_prepare_gym_fastapi_app(res_server), host, res_port))
        running_servers.append(await _start_uvicorn_server(_prepare_gym_fastapi_app(model_server), host, model_port))
        running_servers.append(await _start_uvicorn_server(_prepare_gym_fastapi_app(agent_server), host, agent_port))

        verify_results: list[dict[str, Any]] = []
        t_suite_start = time.perf_counter()
        for idx, task_row in enumerate(tasks):
            payload = dict(task_row)
            payload.setdefault("_ng_task_index", idx)
            payload.setdefault("_ng_rollout_index", 0)
            resp = await server_client.post(
                server_name="sherlock_agent",
                url_path="/run",
                json=payload,
            )
            verified = await resp.json()
            verified["_ng_task_index"] = payload["_ng_task_index"]
            verified["_ng_rollout_index"] = payload["_ng_rollout_index"]
            verify_results.append(verified)
        suite_elapsed_ms = (time.perf_counter() - t_suite_start) * 1000.0

        agg_resp = await server_client.post(
            server_name="sherlock",
            url_path="/aggregate_metrics",
            json={"verify_responses": verify_results},
        )
        agg_payload = await agg_resp.json()
    finally:
        await _stop_uvicorn_servers(running_servers)

    n = max(1, len(verify_results))
    avg_tool_validity = sum(float(r["tool_call_validity"]) for r in verify_results) / n
    avg_task_success = sum(float(r["task_success"]) for r in verify_results) / n
    avg_steps = sum(float(r["steps_to_report"]) for r in verify_results) / n
    avg_time_ms = sum(float(r["time_to_report_ms"]) for r in verify_results) / n
    avg_action_correct = sum(float(r["action_correct"]) for r in verify_results) / n
    avg_recovery = sum(float(r["recovery_score"]) for r in verify_results) / n
    avg_benign_no_snap = sum(float(r["benign_no_snapshot"]) for r in verify_results) / n
    avg_weapon_protocol = sum(float(r["weapon_protocol_score"]) for r in verify_results) / n
    avg_vlm_calls = sum(float(r["vlm_calls"]) for r in verify_results) / n
    avg_reward = sum(float(r["reward"]) for r in verify_results) / n

    enable_thinking = bool((model_block.get("chat_template_kwargs") or {}).get("enable_thinking", False))

    per_scene = [
        {
            "scene_id": r["scene_id"],
            "category": r["category"],
            "task_success": r["task_success"],
            "tool_call_validity": r["tool_call_validity"],
            "steps_to_report": r["steps_to_report"],
            "time_to_report_ms": r["time_to_report_ms"],
            "action_correct": r["action_correct"],
            "recovery_score": r["recovery_score"],
            "benign_no_snapshot": r["benign_no_snapshot"],
            "weapon_protocol_score": r["weapon_protocol_score"],
            "vlm_calls": r["vlm_calls"],
            "reward": r["reward"],
            "report": r.get("report_summary") or (r.get("info") or {}).get("report"),
        }
        for r in verify_results
    ]

    artifact: dict[str, Any] = {
        "ablation_id": agent_config.ablation_id,
        "config_path": str(Path(config_path)),
        "ablation_component": agent_config.ablation_component,
        "ablation_description": agent_config.ablation_description,
        "execution_mode": "local_stub" if effective_use_stub else "live_endpoint",
        "director_base_url": "local_stub" if effective_use_stub else env_director_url,
        "agent_components": {
            "director_model": model_block["model"],
            "director_params_b": agent_config.director_params_b,
            "vlm_model": agent_config.vlm_model_name,
            "vlm_params_b": agent_config.vlm_params_b,
            "total_generative_params_b": round(total_params_b, 2),
            "within_33b_budget": total_params_b <= 33.0,
            "enable_thinking": enable_thinking,
            "prompt_config": agent_config.prompt_config,
            "tool_descriptions": agent_config.tool_descriptions,
            "max_steps": agent_config.max_steps,
        },
        "summary_metrics": {
            "num_tasks": len(verify_results),
            "tool_call_validity": round(avg_tool_validity, 4),
            "task_success": round(avg_task_success, 4),
            "steps_to_report": round(avg_steps, 2),
            "time_to_report_ms": round(avg_time_ms, 2),
            "action_correct": round(avg_action_correct, 4),
            "recovery_score": round(avg_recovery, 4),
            "benign_no_snapshot": round(avg_benign_no_snap, 4),
            "weapon_protocol_score": round(avg_weapon_protocol, 4),
            "avg_vlm_calls": round(avg_vlm_calls, 2),
            "reward": round(avg_reward, 4),
            "suite_wall_time_ms": round(suite_elapsed_ms, 2),
        },
        "nemo_gym_aggregate_metrics": {
            "key_metrics": agg_payload.get("key_metrics", {}),
            "agent_metrics": agg_payload.get("agent_metrics", {}),
        }
        if isinstance(agg_payload, dict)
        else {},
        "per_scene": per_scene,
    }

    out_dir = REPO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_{agent_config.ablation_id}.json"
    out_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def format_markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Ablation Config | Director Model | Total Gen Params | Prompt | Tools | Thinking | Tool-Call Validity | Task Success | Steps to Report | Time to Report (ms) |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        comp = r["agent_components"]
        m = r["summary_metrics"]
        short_model = comp["director_model"].split("/")[-1]
        short_prompt = Path(comp["prompt_config"]).stem
        thinking_str = "on" if comp["enable_thinking"] else "off"
        lines.append(
            f"| `{r['ablation_id']}` | `{short_model}` | {comp['total_generative_params_b']:.2f}B "
            f"| `{short_prompt}` | `{comp['tool_descriptions']}` | `{thinking_str}` "
            f"| {m['tool_call_validity'] * 100:.1f}% | {m['task_success'] * 100:.1f}% "
            f"| {m['steps_to_report']:.2f} | {m['time_to_report_ms']:.1f} |"
        )
    return "\n".join(lines)


async def _async_main(args: argparse.Namespace) -> None:
    config_paths = DEFAULT_CONFIG_ORDER if args.all else [args.config]
    results: list[dict[str, Any]] = []
    for cfg_path in config_paths:
        res = await run_single_ablation(
            config_path=cfg_path,
            use_stub=True if args.stub else None,
            output_dir=args.output_dir,
        )
        results.append(res)

    if args.all or len(results) > 1:
        out_dir = REPO_ROOT / args.output_dir
        summary_path = out_dir / "ablation_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "framework": "NeMo Gym v0.7.0 (native resources_server + responses_api_agent + vllm_model)",
                    "execution_mode": results[0]["execution_mode"],
                    "num_configs": len(results),
                    "num_scenes_per_config": results[0]["summary_metrics"]["num_tasks"],
                    "results": [
                        {
                            "ablation_id": r["ablation_id"],
                            "config_path": r["config_path"],
                            "ablation_component": r["ablation_component"],
                            "agent_components": r["agent_components"],
                            "summary_metrics": r["summary_metrics"],
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
            + "\n"
        )

    print(format_markdown_table(results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sherlock NeMo Gym agent component ablations.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=str,
        help="Path to a single ablation YAML config (e.g. gym/configs/ablation_nemotron_lightning.yaml)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all 5 Sherlock YAML ablation configurations and write results/ablation_*.json",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Force local OpenAI-compatible stub model server even if DIRECTOR_BASE_URL is set",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output JSON files (default: results)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(_async_main(args))
    finally:
        global_aiohttp_client_exit()


if __name__ == "__main__":
    main()
