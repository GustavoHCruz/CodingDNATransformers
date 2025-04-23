from enum import Enum

from models.base_model import BaseModel
from sqlmodel import Field


class SourceEnum(str, Enum):
  genbank = "genbank"
  gencode = "gencode"

class ExInTranslator(BaseModel, table=True):
  hash_id: str = Field(unique=True, index=True)
  source: SourceEnum
  dataset_id: int
  parent_id: int | None
  sequence: str
  organism: str | None
  target_sequence: str