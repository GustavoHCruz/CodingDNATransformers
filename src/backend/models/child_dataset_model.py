from typing import List, Optional

from models.base_model import BaseModel
from models.generation_batch_model import GenerationBatch
from models.parent_dataset_model import ParentDataset
from sqlmodel import Field, Relationship


class ChildDataset(BaseModel, table=True):
  batch_id: int = Field(foreign_key="generationbatch.id", index=True)
  parent_id: int = Field(foreign_key="parentdataset.id", index=True)
  name: str
  record_count: Optional[int] = 0

  batch: GenerationBatch = Relationship(back_populates="children")
  parent: ParentDataset = Relationship()
  assignments: List["ChildRecord"] = Relationship(back_populates="child")