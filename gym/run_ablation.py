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

from gym import ensure_nemo_gym_usage_compat, raise_missing_nemo_gym
from gym.stub_model_server import create_stub_app

try:
    import nemo_gym.server_utils as ng_server_utils
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
except ModuleNotFoundError as _ng_exc:
    raise_missing_nemo_gym(_ng_exc, cli=(__name__ == "__main__"))

ensure_nemo_gym_usage_compat()

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_ORDER = [
    "gym/configs/ablation_prompt_v1.yaml",
    "gym/configs/ablation_terse_tools.yaml",
    "gym/configs/ablation_thinking_on.yaml",
    "gym/configs/ablation_nemotron_lightning.yaml",
    "gym/configs/ablation_qwen_director.yaml",
]


def _resolve_config_file(config_path: str | Path) -> Path:
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        cfg_file = REPO_ROOT / config_path
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Ablation config not found: {config_path}")
    return cfg_file


def get_config_endpoint_info(config_path: str | Path) -> tuple[str, str]:
    """Return (ablation_id, endpoint_env_var) declared in an ablation YAML config."""
    cfg_file = _resolve_config_file(config_path)
    raw_cfg = OmegaConf.load(cfg_file)
    agent_cfg = raw_cfg.sherlock_agent.responses_api_agents.sherlock_agent
    ablation_id = str(getattr(agent_cfg, "ablation_id", cfg_file.stem))
    endpoint_env_var = str(getattr(agent_cfg, "endpoint_env_var", "DIRECTOR_BASE_URL"))
    return ablation_id, endpoint_env_var


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


def _validate_agent_run_response(
    verified: Any,
    status_code: int,
    scene_id: str,
    ablation_id: str,
) -> dict[str, Any]:
    """Validate the response returned by sherlock_agent /run and fail with a clear error."""
    if status_code >= 400 or not isinstance(verified, dict) or "task_success" not in verified:
        raise RuntimeError(
            f"Agent '/run' failed on scene {scene_id!r} for ablation "
            f"{ablation_id!r} (HTTP {status_code}): {verified}"
        )
    return verified


async def run_single_ablation(
    config_path: str | Path,
    use_stub: bool | None = None,
    output_dir: str | Path = "results",
) -> dict[str, Any]:
    """Run one NeMo Gym YAML ablation configuration end-to-end and save JSON results."""
    ensure_nemo_gym_usage_compat()
    cfg_file = _resolve_config_file(config_path)

    raw_cfg = OmegaConf.load(cfg_file)
    resolved_dict: dict[str, Any] = OmegaConf.to_container(raw_cfg, resolve=True)  # type: ignore[assignment]

    agent_block = resolved_dict["sherlock_agent"]["responses_api_agents"]["sherlock_agent"]
    endpoint_env_var = str(agent_block.get("endpoint_env_var") or "DIRECTOR_BASE_URL")
    env_director_url = os.environ.get(endpoint_env_var, "").strip()

    if use_stub is None:
        effective_use_stub = not bool(env_director_url)
    else:
        effective_use_stub = use_stub

    if not effective_use_stub and not env_director_url:
        raise RuntimeError(
            f"Ablation {agent_block.get('ablation_id')!r} ({config_path}) requires "
            f"environment variable {endpoint_env_var} to be set for live execution."
        )

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
    default_model_id = (
        "Qwen/Qwen2.5-7B-Instruct"
        if endpoint_env_var == "QWEN_DIRECTOR_BASE_URL"
        else "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    )
    if not str(model_block.get("model") or "").strip():
        model_block["model"] = default_model_id
    if not str(agent_block.get("director_model_name") or "").strip():
        agent_block["director_model_name"] = model_block["model"]
    if effective_use_stub:
        model_block["base_url"] = f"http://{host}:{stub_port}/v1"
        model_block["api_key"] = "EMPTY"
    elif env_director_url:
        model_block["base_url"] = env_director_url

    agent_block["host"] = host
    agent_block["port"] = agent_port

    global_config_dict = OmegaConf.create(resolved_dict)
    head_server_config = BaseServerConfig(host=host, port=res_port)
    server_client = ServerClient(
        head_server_config=head_server_config,
        global_config_dict=global_config_dict,
    )

    existing_client = getattr(ng_server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
    if existing_client is not None and (
        existing_client.closed
        or getattr(existing_client, "_loop", None) is not asyncio.get_running_loop()
    ):
        ng_server_utils._GLOBAL_AIOHTTP_CLIENT = None
        ng_server_utils._GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY = False
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
            raw_text = await resp.text()
            try:
                raw_verified = json.loads(raw_text)
            except (json.JSONDecodeError, TypeError):
                raw_verified = raw_text
            verified = _validate_agent_run_response(
                verified=raw_verified,
                status_code=getattr(resp, "status", 200),
                scene_id=str(payload.get("scene_id", f"task_{idx}")),
                ablation_id=agent_config.ablation_id,
            )
            verified["_ng_task_index"] = payload["_ng_task_index"]
            verified["_ng_rollout_index"] = payload["_ng_rollout_index"]
            verify_results.append(verified)
        suite_elapsed_ms = (time.perf_counter() - t_suite_start) * 1000.0

        agg_resp = await server_client.post(
            server_name="sherlock",
            url_path="/aggregate_metrics",
            json={"verify_responses": verify_results},
        )
        agg_text = await agg_resp.text()
        try:
            agg_payload = json.loads(agg_text)
        except (json.JSONDecodeError, TypeError):
            agg_payload = agg_text
        if getattr(agg_resp, "status", 200) >= 400 or not isinstance(agg_payload, dict):
            raise RuntimeError(
                f"Resources server '/aggregate_metrics' failed for ablation "
                f"{agent_config.ablation_id!r} (HTTP {getattr(agg_resp, 'status', 200)}): {agg_payload}"
            )
    finally:
        await _stop_uvicorn_servers(running_servers)
        active_client = getattr(ng_server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
        if active_client is not None and not active_client.closed:
            await active_client.close()
        ng_server_utils._GLOBAL_AIOHTTP_CLIENT = None
        ng_server_utils._GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY = False

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

    if effective_use_stub:
        model_server_meta: dict[str, Any] = {
            "name": "local_stub",
            "version": None,
            "reachable": True,
            "source": "stub",
            "host": "laptop",
            "label": "local_stub",
        }
        runtime_label = "local_stub"
    else:
        from sherlock.config import Endpoint
        from sherlock.dashboard.model_servers import clear_cache, server

        clear_cache()
        model_server_meta = server(
            "director",
            Endpoint(
                base_url=env_director_url,
                model=str(model_block["model"]),
                api_key=str(model_block.get("api_key") or "EMPTY"),
            ),
        )
        detected_name = str(model_server_meta.get("name") or "").strip()
        host_name = str(model_server_meta.get("host") or "laptop").strip()
        if detected_name:
            prefix = "local" if host_name == "laptop" else host_name
            runtime_label = f"{prefix}_{detected_name.lower()}"
        else:
            runtime_label = f"{host_name}_endpoint"

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
        "endpoint_env_var": endpoint_env_var,
        "execution_mode": "local_stub" if effective_use_stub else "live_endpoint",
        "runtime": runtime_label,
        "model_server": model_server_meta,
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
            "endpoint_env_var": endpoint_env_var,
            "model_server": model_server_meta.get("label"),
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

    out_dir = Path(output_dir) if Path(output_dir).is_absolute() else REPO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = "ablation_stub" if effective_use_stub else "ablation"
    out_path = out_dir / f"{file_prefix}_{agent_config.ablation_id}.json"
    out_path.write_text(json.dumps(artifact, indent=2) + "\n")
    if not effective_use_stub and (out_dir / "ablation_summary.json").is_file():
        _write_summary_file(out_dir, [artifact], [])
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


def _summary_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "ablation_id": r["ablation_id"],
        "config_path": r["config_path"],
        "ablation_component": r["ablation_component"],
        "endpoint_env_var": r.get("endpoint_env_var", "DIRECTOR_BASE_URL"),
        "runtime": r.get("runtime"),
        "model_server": r.get("model_server"),
        "agent_components": r["agent_components"],
        "summary_metrics": r["summary_metrics"],
    }


def _write_summary_file(
    out_dir: Path,
    results: list[dict[str, Any]],
    skipped: list[dict[str, str]],
) -> None:
    if not results:
        return
    mode = results[0]["execution_mode"]
    is_stub = mode == "local_stub"
    file_prefix = "ablation_stub" if is_stub else "ablation"
    summary_name = "ablation_stub_summary.json" if is_stub else "ablation_summary.json"
    summary_path = out_dir / summary_name

    by_id: dict[str, dict[str, Any]] = {r["ablation_id"]: r for r in results}
    for cfg_path in DEFAULT_CONFIG_ORDER:
        ablation_id, _ = get_config_endpoint_info(cfg_path)
        if ablation_id in by_id:
            continue
        existing_path = out_dir / f"{file_prefix}_{ablation_id}.json"
        if not existing_path.is_file():
            continue
        try:
            existing = json.loads(existing_path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(existing, dict) and existing.get("execution_mode") == mode:
            by_id[ablation_id] = existing

    ordered_ids = [get_config_endpoint_info(p)[0] for p in DEFAULT_CONFIG_ORDER]
    merged_results = [by_id[aid] for aid in ordered_ids if aid in by_id] + [
        r for aid, r in by_id.items() if aid not in ordered_ids
    ]

    summary_path.write_text(
        json.dumps(
            {
                "framework": "NeMo Gym v0.7.0 (native resources_server + responses_api_agent)",
                "execution_mode": mode,
                "runtime": results[0].get("runtime"),
                "model_server": results[0].get("model_server"),
                "num_configs": len(merged_results),
                "num_skipped": len(skipped),
                "skipped_configs": skipped,
                "num_scenes_per_config": results[0]["summary_metrics"]["num_tasks"],
                "results": [_summary_row(r) for r in merged_results],
            },
            indent=2,
        )
        + "\n"
    )


async def run_all_ablations(
    config_paths: list[str] | None = None,
    force_stub: bool = False,
    output_dir: str | Path = "results",
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Run all configured ablations, skipping any config whose endpoint_env_var is unset in live mode."""
    paths = list(config_paths) if config_paths is not None else list(DEFAULT_CONFIG_ORDER)
    endpoint_infos = [(cfg_path, *get_config_endpoint_info(cfg_path)) for cfg_path in paths]

    known_env_vars = {get_config_endpoint_info(p)[1] for p in DEFAULT_CONFIG_ORDER} | {
        env_var for _, _, env_var in endpoint_infos
    }
    any_endpoint_set = any(
        bool(os.environ.get(env_var, "").strip()) for env_var in known_env_vars
    )
    live_mode = not force_stub and any_endpoint_set

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for cfg_path, ablation_id, endpoint_env_var in endpoint_infos:
        endpoint_url = os.environ.get(endpoint_env_var, "").strip()
        if live_mode and not endpoint_url:
            print(
                f"Skipping ablation '{ablation_id}' ({cfg_path}): "
                f"endpoint env var {endpoint_env_var} is not set.",
                flush=True,
            )
            skipped.append(
                {
                    "ablation_id": ablation_id,
                    "config_path": str(cfg_path),
                    "endpoint_env_var": endpoint_env_var,
                    "reason": f"{endpoint_env_var} is not set",
                }
            )
            continue

        res = await run_single_ablation(
            config_path=cfg_path,
            use_stub=not live_mode,
            output_dir=output_dir,
        )
        results.append(res)

    out_dir = Path(output_dir) if Path(output_dir).is_absolute() else REPO_ROOT / output_dir
    _write_summary_file(out_dir, results, skipped)
    return results, skipped


async def _async_main(args: argparse.Namespace) -> None:
    if args.all:
        results, _skipped = await run_all_ablations(
            force_stub=bool(args.stub),
            output_dir=args.output_dir,
        )
    else:
        res = await run_single_ablation(
            config_path=args.config,
            use_stub=True if args.stub else None,
            output_dir=args.output_dir,
        )
        results = [res]

    if results:
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
