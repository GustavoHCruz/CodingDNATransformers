from typing import Optional

from models.base_model import ApproachEnum
from models.parent_dataset_model import OriginEnum
from pydantic import BaseModel


class ProcessedDatasetCreation(BaseModel):
  name: str
  parent_id: int
  approach: ApproachEnum
  origin: OriginEnum
  sizes: list[int]
  seed: Optional[int] = 1234