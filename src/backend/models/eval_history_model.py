from datetime import time

from models.base_model import BaseModel
from sqlmodel import Field


class EvalHistory(BaseModel, table=True):
  model_id: int = Field(foreign_key="modelhistory.id", index=True)
  loss: float
  acc: float
  duration: time