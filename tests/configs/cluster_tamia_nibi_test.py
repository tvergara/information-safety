from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

CLUSTER_DIR = Path(__file__).parent.parent.parent / "information_safety" / "configs" / "cluster"


@pytest.mark.parametrize("cluster_name", ["tamia", "nibi"])
def test_cluster_yaml_loads(cluster_name: str) -> None:
    yaml_path = CLUSTER_DIR / f"{cluster_name}.yaml"
    assert yaml_path.exists(), f"{yaml_path} should exist"
    cfg = OmegaConf.load(yaml_path)
    assert isinstance(cfg, DictConfig)
    defaults = OmegaConf.to_container(cfg.defaults)
    assert isinstance(defaults, list)
    assert "mila.yaml" in defaults
    executor = cfg.hydra.launcher.executor
    assert executor.cluster_hostname == cluster_name
    assert executor.internet_access_on_compute_nodes is False
