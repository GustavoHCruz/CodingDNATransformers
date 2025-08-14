from dataclasses import dataclass


@dataclass
class TrainParams:
  epochs: int
  batch_size: int
  gradient_accumulation: int
  lr: float
  warmup_ratio: float