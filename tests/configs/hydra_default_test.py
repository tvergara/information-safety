from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).parent.parent.parent / "information_safety" / "configs"


def test_hydra_default_run_and_sweep_dirs_root_under_scratch() -> None:
    """hydra/default.yaml must root run/sweep dirs under $SCRATCH so Lightning checkpoints don't
    fill HOME quota."""
    data = yaml.safe_load((CONFIGS_DIR / "hydra" / "default.yaml").read_text())
    expected_prefix = "${oc.env:SCRATCH,.}/information-safety/logs/"
    assert data["run"]["dir"].startswith(expected_prefix)
    assert data["sweep"]["dir"].startswith(expected_prefix)
