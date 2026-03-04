"""Configuration dataclasses for model and tokenizer loading."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

logger = getLogger(__name__)


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for loading a causal LM.

    Supports both `from_pretrained` (default) and `from_config` (random init) modes.
    """

    pretrained_model_name_or_path: str
    trust_remote_code: bool = False
    torch_dtype: str | None = None
    attn_implementation: str | None = None
    from_config: bool = False


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for loading a tokenizer."""

    pretrained_model_name_or_path: str
    trust_remote_code: bool = False
    use_fast: bool = True


def load_model(config: NetworkConfig) -> PreTrainedModel:
    """Load a causal LM model from a NetworkConfig.

    If `config.from_config` is True, initializes with random weights (useful for
    comparing pretrained vs random init). Otherwise loads pretrained weights.
    """
    kwargs: dict[str, object] = {
        "trust_remote_code": config.trust_remote_code,
    }
    if config.torch_dtype is not None:
        kwargs["torch_dtype"] = getattr(torch, config.torch_dtype)
    if config.attn_implementation is not None:
        kwargs["attn_implementation"] = config.attn_implementation

    if config.from_config:
        logger.info(
            f"Initializing model with random weights from config: "
            f"{config.pretrained_model_name_or_path}"
        )
        model_config = AutoConfig.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        return AutoModelForCausalLM.from_config(model_config, **kwargs)

    logger.info(f"Loading pretrained model: {config.pretrained_model_name_or_path}")
    return AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name_or_path, **kwargs
    )
