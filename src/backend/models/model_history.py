from datetime import time
from typing import List, Optional

from models.base_model import ApproachEnum, BaseModel
from sqlmodel import Relationship


class ModelHistory(BaseModel, table=True):
  name: str
  approach: ApproachEnum
  model: str
  path: str
  seed: int
  epochs: int
  lr: int
  batch_size: int
  acc: float
  duration: time
  hide_prob: Optional[float] = None

  train_history: List["TrainHistory"] = Relationship(back_populates="model")