from datetime import time

from models.base_model import BaseModel
from models.model_history import ModelHistory
from sqlmodel import Field, Relationship


class TrainHistory(BaseModel, table=True):
  model_id: int = Field(foreign_key="modelhistory.id", index=True)
  epoch: int
  loss: float
  duration: time

  model: ModelHistory = Relationship(back_populates="train_history")