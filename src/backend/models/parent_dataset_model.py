from enum import Enum
from typing import List, Optional

from models.base_model import ApproachEnum, BaseModel
from sqlmodel import Relationship


class OriginEnum(str, Enum):
  genbank = "genbank"
  gencode = "gencode"


class ParentDataset(BaseModel, table=True):
  name: str
  approach: ApproachEnum
  origin: OriginEnum
  record_count: Optional[int] = 0

  records: List["ParentRecord"] = Relationship(back_populates="parent")