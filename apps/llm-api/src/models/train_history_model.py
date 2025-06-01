from datetime import time

from sqlmodel import Field, Relationship

from models.base_model import BaseModel
from models.model_history_model import ModelHistory


class TrainHistory(BaseModel, table=True):
  model_id: int = Field(foreign_key="modelhistory.id", index=True)
  epoch: int
  loss: float
  duration: time

  model: ModelHistory = Relationship(back_populates="train_history")