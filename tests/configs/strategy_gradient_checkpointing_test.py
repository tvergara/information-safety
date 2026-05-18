"""Each strategy YAML must declare `gradient_checkpointing` so Hydra's struct mode accepts
`algorithm.strategy.gradient_checkpointing=...` overrides emitted by `scripts/build_job_queue.py`
for gpt-oss-20b jobs."""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

STRATEGY_DIR = (
    Path(__file__).parent.parent.parent
    / "information_safety"
    / "configs"
    / "algorithm"
    / "strategy"
)


@pytest.mark.parametrize("yaml_name", ["lora.yaml", "data.yaml", "data-deferred.yaml"])
def test_strategy_yaml_accepts_gradient_checkpointing_override(yaml_name: str) -> None:
    cfg = OmegaConf.load(STRATEGY_DIR / yaml_name)
    assert isinstance(cfg, DictConfig)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.update(cfg, "gradient_checkpointing", True)
    assert cfg.gradient_checkpointing is True
