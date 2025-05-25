from datetime import datetime
from typing import Optional

from models.progress_tracker_model import ProgressTypeEnum, StatusEnum
from pydantic import BaseModel


class ProgressTrackerGetDTO(BaseModel):
  id: int
  task_name: str
  progress: float
  progress_type: ProgressTypeEnum
  status: StatusEnum
  info: Optional[str] = ""
  created_at: datetime

  class Config:
    from_attributes = True