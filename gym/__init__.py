"""NeMo Gym ablation environment, agent servers, and configs for Sherlock."""
from __future__ import annotations

import sys
from typing import NoReturn

MIN_GYM_PYTHON = (3, 13, 14)


def format_missing_nemo_gym_error(version_info: tuple[int, int, int] | None = None) -> str:
    """Return a clear installation hint when nemo_gym is missing or Python < 3.13.14."""
    ver = version_info if version_info is not None else sys.version_info[:3]
    py_ver = f"{ver[0]}.{ver[1]}.{ver[2]}"
    if ver < MIN_GYM_PYTHON:
        return (
            f"NeMo Gym requires Python >= 3.13.14 (current Python is {py_ver}); "
            "install the gym extra on Python >= 3.13.14: "
            "uv run --python 3.13.14 --extra gym python -m gym.run_ablation --all --stub"
        )
    return (
        f"nemo-gym is not installed in the active environment (Python {py_ver}); "
        "install the gym extra on Python >= 3.13.14: "
        "uv sync --extra gym (or run with `uv run --extra gym python -m gym.run_ablation ...`)."
    )


def raise_missing_nemo_gym(exc: ModuleNotFoundError, *, cli: bool = False) -> NoReturn:
    """Raise a clear error when nemo_gym is unavailable."""
    msg = format_missing_nemo_gym_error()
    if cli:
        raise SystemExit(msg) from exc
    raise ModuleNotFoundError(msg, name="nemo_gym") from exc


def ensure_nemo_gym_usage_compat() -> None:
    """Ensure NeMo Gym token-details models accept optional SDK fields (e.g. cache_write_tokens).

    In nemo-gym 0.7.0, responses_converter.py and NeMoGymResponseUsage.sum_from_list construct
    NeMoGymResponseInputTokensDetails(cached_tokens=...) without passing newer OpenAI SDK fields
    such as cache_write_tokens (which openai>=3 marks required on InputTokensDetails). Relaxing
    any extra inherited required fields on NeMoGymResponseInputTokensDetails and
    NeMoGymResponseOutputTokensDetails to Optional[int] = None ensures both the local stub and
    live vLLM 0.28 endpoints validate cleanly across OpenAI SDK versions.
    """
    try:
        from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
        from nemo_gym.openai_utils import (
            NeMoGymResponse,
            NeMoGymResponseInputTokensDetails,
            NeMoGymResponseOutputTokensDetails,
            NeMoGymResponseUsage,
        )
    except ImportError:
        return

    rebuilt = False
    for details_cls, primary_field in (
        (NeMoGymResponseInputTokensDetails, "cached_tokens"),
        (NeMoGymResponseOutputTokensDetails, "reasoning_tokens"),
    ):
        cls_changed = False
        for field_name, field_info in details_cls.model_fields.items():
            if field_name != primary_field and field_info.is_required():
                field_info.default = None
                field_info.annotation = int | None
                cls_changed = True
        if cls_changed:
            details_cls.model_rebuild(force=True)
            rebuilt = True

    if rebuilt:
        NeMoGymResponseUsage.model_rebuild(force=True)
        NeMoGymResponse.model_rebuild(force=True)
        BaseVerifyRequest.model_rebuild(force=True)
        BaseVerifyResponse.model_rebuild(force=True)


ensure_nemo_gym_usage_compat()
