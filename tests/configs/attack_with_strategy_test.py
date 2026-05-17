from pathlib import Path

import yaml

CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / "information_safety"
    / "configs"
    / "algorithm"
    / "attack_with_strategy.yaml"
)


def test_network_config_interpolates_torch_dtype_from_model_group() -> None:
    """Without this interpolation, model groups' torch_dtype is ignored and
    AutoModelForCausalLM.from_pretrained() defaults to fp32, doubling GPU memory for the base model
    and OOM-ing 8B+ DataStrategy runs on 80 GB GPUs."""
    data = yaml.safe_load(CONFIG_PATH.read_text())
    network_config = data["network_config"]
    assert network_config["torch_dtype"] == "${..model.torch_dtype}"
