from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Progress(SQLModel, table=True):
  id: str = Field(primary_key=True)
  task_name: str
  progress: float = 0.0
  status: str = "in_progress"
  info: Optional[str] = None
  updated_at: datetime = Field(default_factory=datetime.now)