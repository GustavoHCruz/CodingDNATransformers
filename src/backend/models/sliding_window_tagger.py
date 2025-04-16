from enum import Enum

from models.base_model import BaseModel
from sqlmodel import Field


class SourceEnum(str, Enum):
  genbank = "genbank"
  gencode = "gencode"

class SlidingWindowTagger(BaseModel, table=True):
  parent_id: int | None
  source: SourceEnum
  sequence: str
  organism: str | None
  target_sequence: str
  hash_id: str = Field(unique=True, index=True)