from enum import Enum
from typing import Optional

from models.base_model import BaseModel


class ProgressTypeEnum(str, Enum):
  percentage = "percentage"
  counter = "counter"

class StatusEnum(str, Enum):
  in_progress = "in_progress"
  complete = "complete"
  failed = "failed"

class ProgressTracker(BaseModel, table=True):
  task_name: Optional[str] = None
  progress: float = 0.0
  progress_type: ProgressTypeEnum
  status: StatusEnum = StatusEnum.in_progress
  info: Optional[str] = None