from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainParams:
  batch_size: int = 4
  gradient_accumulation: int = 1
  lr: float = 5e-5
  epochs: int = 3
  warmup_ratio: int | None = None
  optim: Literal[
    "adamw_torch",
    "sgd",
  ] = "adamw_torch"
  logging_steps: int = 2000