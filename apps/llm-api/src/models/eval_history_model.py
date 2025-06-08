from datetime import time

from models.base_model import BaseModel
from models.model_history_model import ModelHistory
from sqlmodel import Field, Relationship


class EvalHistory(BaseModel, table=True):
  model_id: int = Field(foreign_key="modelhistory.id", index=True)
  loss: float
  acc: float
  duration: time

  model: ModelHistory = Relationship(back_populates="eval_history")