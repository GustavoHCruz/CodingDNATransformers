from typing import Optional

from models.base_model import BaseModel


class Progress(BaseModel, table=True):
  task_name: str
  progress: float = 0.0
  status: str = "in_progress"
  info: Optional[str] = None