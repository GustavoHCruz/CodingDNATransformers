from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class ApproachEnum(str, Enum):
  exin_classifier = "exin_classifier"
  exin_translator = "exin_translator"
  protein_translator = "protein_translator"
  sliding_window_extraction = "sliding_window_extraction"

class Datasets(SQLModel, table=True):
  id: Optional[int] = Field(default=None, primary_key=True)
  approach: ApproachEnum
  name: str
  sequence: str
  organism: str
  target_sequence: str
  hash_id: str = Field(unique=True, index=True)