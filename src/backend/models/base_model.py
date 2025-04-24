from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class ApproachEnum(str, Enum):
  exin_classifier = "exin_classifier"
  exin_translator = "exin_translator"
  protein_translator = "protein_translator"
  sliding_window_extraction = "sliding_window_extraction"

class BaseModel(SQLModel):
  id: Optional[int] = Field(default=None, primary_key=True)
  created_at: datetime = Field(default_factory=datetime.now, nullable=False)
  updated_at: datetime = Field(default_factory=datetime.now, nullable=False)

  @staticmethod
  def _update_timestamp(mapper, connection, target) -> None:
    target.updated_at = datetime.now()

class MinimalModel(SQLModel):
  id: Optional[int] = Field(default=None, primary_key=True)