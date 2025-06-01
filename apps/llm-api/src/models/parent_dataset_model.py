from enum import Enum
from typing import List, Optional

from sqlmodel import Relationship

from models.base_model import ApproachEnum, BaseModel


class OriginEnum(str, Enum):
  genbank = "genbank"
  gencode = "gencode"


class ParentDataset(BaseModel, table=True):
  name: str
  approach: ApproachEnum
  origin: OriginEnum
  record_count: Optional[int] = 0

  records: List["ParentRecord"] = Relationship(back_populates="parent")