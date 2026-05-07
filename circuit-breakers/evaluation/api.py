from dataclasses import dataclass

from numpy import ndarray


@dataclass
class EvalInstance:
    behavior: str
    context: str | None = None
    default_target: str | None = None
    activation_norms: ndarray | None = None
    final_loss: float | None = None
    final_string: str | None = None
    generation: str | None = None
    input_embeds: list | None = None
    losses: list | None = None
    messages: str | None = None
    method: str | None = None
    score: int | None = None
    tokens: ndarray | None = None
