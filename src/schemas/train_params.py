from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainParams:
  batch_size: int
  gradient_accumulation: int
  lr: float
  epochs: int
  optim: Literal[
    "adamw_torch",
    "sgd",
  ] = "adamw_torch"
  logging_steps: int = 2000