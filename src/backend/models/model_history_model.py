from __future__ import annotations

from datetime import timedelta
from typing import List, Optional

from models.base_model import ApproachEnum, BaseModel
from sqlmodel import Field, Relationship


class ModelHistory(BaseModel, table=True):
  parent_model_id: Optional[int] = Field(foreign_key="modelhistory.id", index=True)
  name: str
  approach: ApproachEnum
  model: str
  path: str
  seed: int
  epochs: int
  lr: float
  batch_size: int
  acc: float
  duration: timedelta
  hide_prob: Optional[float] = None

  model: Optional[ModelHistory] = Relationship(back_populates="train_history")
  train_history: List["TrainHistory"] = Relationship(back_populates="model")