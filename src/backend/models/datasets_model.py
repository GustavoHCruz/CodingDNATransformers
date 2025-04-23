from enum import Enum
from typing import Optional

from models.base_model import BaseModel


class ApproachEnum(str, Enum):
  exin_classifier = "exin_classifier"
  exin_translator = "exin_translator"
  protein_translator = "protein_translator"
  sliding_window_extraction = "sliding_window_extraction"

class DatasetTypeEnum(str, Enum):
  base = "base"
  child = "child"

class OriginEnum(str, Enum):
  genbank = "genbank"
  gencode = "gencode"

class Datasets(BaseModel, table=True):
  approach: ApproachEnum
  origin: OriginEnum
  name: str
  dataset_type: DatasetTypeEnum = DatasetTypeEnum.base
  parent_id: Optional[int] = None