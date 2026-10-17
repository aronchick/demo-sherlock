"""Native NeMo Gym Responses API Agent for Sherlock Director ablations.

All ablated agent components (model server, system prompt YAML, tool description
verbosity, max steps) are configured via YAML.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, Request, Response
from nemo_gym.base_resources_server import BaseSeedSessionResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from omegaconf import OmegaConf
from pydantic import PrivateAttr

from gym.resources_servers.sherlock.app import (
    SherlockRunRequest,
    SherlockSeedSessionRequest,
    SherlockVerifyRequest,
    SherlockVerifyResponse,
)
from gym.resources_servers.sherlock.scenes import get_tool_schemas

_DEFAULT_RESPONSES_BODY = Body()


class SherlockAgentConfig(BaseResponsesAPIAgentConfig):
    entrypoint: str = "app.py"
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = 6
    weapon_extra_steps: int = 4
    prompt_config: str | None = "gym/prompts/system_v2.yaml"
    tool_descriptions: str = "verbose"  # "verbose" | "terse"
    ablation_id: str = "nemotron_lightning"
    ablation_component: str = "baseline"
    ablation_description: str = "Nemotron 3.5 Lightning (thinking off), system_v2, verbose tools"
    endpoint_env_var: str = "DIRECTOR_BASE_URL"
    director_model_name: str = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    director_params_b: float = 30.0
    vlm_model_name: str = "HuggingFaceTB/SmolVLM-Instruct"
    vlm_params_b: float = 2.25


def _resolve_prompt_path(prompt_config: str) -> Path:
    candidate = Path(prompt_config)
    if candidate.is_file():
        return candidate
    repo_root = Path(__file__).resolve().parents[3]
    in_repo = repo_root / prompt_config
    if in_repo.is_file():
        return in_repo
    raise FileNotFoundError(f"Cannot find prompt_config file: {prompt_config!r}")


def _merge_cookies(jar: dict[str, str], response_cookies: Any) -> dict[str, str]:
    for k, v in (response_cookies or {}).items():
        jar[k] = v.value if hasattr(v, "value") else str(v)
    return jar


def _is_weapon_event(messages: list[Any]) -> bool:
    for msg in messages:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role != "user":
            continue
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else ""
        )
        if isinstance(content, str) and (
            '"threat": "weapon"' in content
            or '"threat":"weapon"' in content
            or '"label": "knife"' in content
            or '"label": "scissors"' in content
        ):
            return True
    return False


class SherlockAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: SherlockAgentConfig
    _system_prompt: str | None = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.config.total_generative_params_b > 33.0:
            raise ValueError(
                f"Configuration exceeds 33B total generative parameter budget: "
                f"{self.config.total_generative_params_b:.2f}B > 33.0B"
            )
        if self.config.prompt_config:
            prompt_path = _resolve_prompt_path(self.config.prompt_config)
            loaded = OmegaConf.load(prompt_path)
            raw_prompt = str(loaded.system).strip()
            self._system_prompt = raw_prompt.replace("{max_calls}", str(self.config.max_steps))
        else:
            self._system_prompt = None

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = _DEFAULT_RESPONSES_BODY,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        else:
            body.input = list(body.input)

        # Apply YAML-configured tool descriptions (verbose vs terse)
        configured_tools = get_tool_schemas(
            mode=self.config.tool_descriptions,
            responses_api_format=True,
        )
        body.tools = configured_tools

        # Prepend YAML-configured system prompt if not already present
        has_system = any(getattr(m, "role", None) == "system" for m in body.input)
        if self._system_prompt and not has_system:
            body.input = [
                NeMoGymEasyInputMessage(role="system", content=self._system_prompt),
                *body.input,
            ]

        effective_max_steps = self.config.max_steps + (
            self.config.weapon_extra_steps if _is_weapon_event(body.input) else 0
        )

        steps = 0
        output_accumulator: list[Any] = []
        cookies: dict[str, str] = _merge_cookies({}, request.cookies)
        model_response: NeMoGymResponse | None = None

        while True:
            steps += 1
            res = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=body,
                cookies=cookies,
            )
            raw_model_payload = await res.json()
            if getattr(res, "status", 200) >= 400 or not isinstance(raw_model_payload, dict):
                raise RuntimeError(
                    f"Model server {self.config.model_server.name!r} returned error "
                    f"(HTTP {getattr(res, 'status', 'unknown')}): {raw_model_payload}"
                )
            model_response = NeMoGymResponse.model_validate(raw_model_payload)
            _merge_cookies(cookies, res.cookies)

            output = model_response.output
            output_accumulator.extend(output)
            step_items: list[Any] = list(output)

            all_fn_calls = [item for item in output if item.type == "function_call"]
            if not all_fn_calls:
                break

            reported_done = False
            for tool_call in all_fn_calls:
                try:
                    tool_args = json.loads(tool_call.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {"_raw_corrupted_arguments": tool_call.arguments}

                tool_res = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{tool_call.name}",
                    json=tool_args if isinstance(tool_args, dict) else {"_raw": tool_args},
                    cookies=cookies,
                )
                _merge_cookies(cookies, tool_res.cookies)
                tool_result = await tool_res.json()
                fn_out = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=tool_call.call_id,
                    output=json.dumps(tool_result),
                )
                output_accumulator.append(fn_out)
                step_items.append(fn_out)
                if (
                    tool_call.name == "report"
                    and isinstance(tool_result, dict)
                    and tool_result.get("done") is True
                ):
                    reported_done = True
                    break

            body.input.extend(step_items)
            if reported_done or steps >= effective_max_steps:
                break

        request._cookies = dict(cookies)
        for k, v in cookies.items():
            response.set_cookie(k, v)

        assert model_response is not None
        model_response.output = output_accumulator
        return model_response

    async def run(self, request: Request, body: SherlockRunRequest) -> SherlockVerifyResponse:
        t0 = time.perf_counter()
        cookies: dict[str, str] = _merge_cookies({}, request.cookies)

        seed_req = SherlockSeedSessionRequest(scene_id=body.scene_id)
        seed_res = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=seed_req,
            cookies=cookies,
        )
        BaseSeedSessionResponse.model_validate(await seed_res.json())
        _merge_cookies(cookies, seed_res.cookies)
        request._cookies = dict(cookies)

        response_obj = await self.responses(request, Response(), body.responses_create_params)
        _merge_cookies(cookies, request.cookies)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        verify_request = SherlockVerifyRequest(
            **body.model_dump(exclude={"elapsed_ms"}),
            elapsed_ms=elapsed_ms,
            response=response_obj,
        )
        verify_res = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request,
            cookies=cookies,
        )
        return SherlockVerifyResponse.model_validate(await verify_res.json())


# Attach property on SherlockAgentConfig for parameter budget validation
SherlockAgentConfig.total_generative_params_b = property(  # type: ignore[attr-defined]
    lambda self: float(self.director_params_b) + float(self.vlm_params_b)
)

if __name__ == "__main__":
    SherlockAgent.run_webserver()
