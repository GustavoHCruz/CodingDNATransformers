from typing import List, Optional

from models.base_model import ApproachEnum, BaseModel
from models.generation_batch_model import GenerationBatch
from models.parent_dataset_model import ParentDataset
from sqlmodel import Field, Relationship


class ChildDataset(BaseModel, table=True):
  name: str
  approach: ApproachEnum
  batch_id: int = Field(foreign_key="generationbatch.id", index=True)
  record_count: Optional[int] = 0

  batch: GenerationBatch = Relationship(back_populates="children")
  assignments: List["ChildRecord"] = Relationship(back_populates="child")