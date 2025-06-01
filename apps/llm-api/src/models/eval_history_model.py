from datetime import time

from sqlmodel import Field

from models.base_model import BaseModel


class EvalHistory(BaseModel, table=True):
  model_id: int = Field(foreign_key="modelhistory.id", index=True)
  loss: float
  acc: float
  duration: time