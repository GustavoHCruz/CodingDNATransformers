from enum import Enum

from models.base_model import BaseModel
from sqlmodel import Field


class SourceEnum(str, Enum):
  genbank = "genbank"
  gencode = "gencode"

class ExInClassifier(BaseModel, table=True):
  hash_id: str = Field(unique=True, index=True)
  source: SourceEnum
  dataset_id: int | None
  sequence: str
  flank_left: str | None
  flank_right: str | None
  organism: str | None
  gene: str | None
  label: str