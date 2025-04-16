from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class BaseModel(SQLModel):
  id: Optional[int] = Field(default=None, primary_key=True)
  created_at: datetime = Field(default_factory=datetime.now, nullable=False)
  updated_at: datetime = Field(default_factory=datetime.now, nullable=False)

  @staticmethod
  def _update_timestamp(mapper, connection, target) -> None:
    target.updated_at = datetime.now()