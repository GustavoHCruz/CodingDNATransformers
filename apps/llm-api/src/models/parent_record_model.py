from typing import List, Optional

from sqlmodel import Field, Relationship

from models.base_model import MinimalModel
from models.parent_dataset_model import ParentDataset


class ParentRecord(MinimalModel, table=True):
  parent_id: int = Field(foreign_key="parentdataset.id", index=True)
  sequence: str
  target: str
  flank_before: Optional[str] = None
  flank_after: Optional[str] = None
  organism: Optional[str] = None
  gene: Optional[str] = None

  parent: ParentDataset = Relationship(back_populates="records")
  child_assignments: List["ChildRecord"] = Relationship(back_populates="parent_record")