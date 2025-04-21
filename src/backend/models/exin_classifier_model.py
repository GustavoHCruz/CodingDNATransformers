from enum import Enum

from models.base_model import BaseModel
from sqlmodel import Field


class SourceEnum(str, Enum):
  genbank = "genbank"
  gencode = "gencode"

class ExInClassifier(BaseModel, table=True):
  parent_id: int | None
  source: SourceEnum
  sequence: str
  flank_left: str | None
  flank_right: str | None
  organism: str | None
  gene: str | None
  label: str
  hash_id: str = Field(unique=True, index=True)