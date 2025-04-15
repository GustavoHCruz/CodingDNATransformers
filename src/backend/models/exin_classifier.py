from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class SourceEnum(str, Enum):
  genbank = "genbank"
  gencode = "gencode"

class ExInClassifier(SQLModel, table=True):
  id: Optional[int] = Field(default=None, primary_key=True)
  parent_id: int | None
  source: SourceEnum
  sequence: str
  flank_left: str | None
  flank_right: str | None
  organism: str | None
  gene: str | None
  label: str
  hash_id: str = Field(unique=True, index=True)