"""NeMo Gym ablation environment, agent servers, and configs for Sherlock."""
from __future__ import annotations


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
