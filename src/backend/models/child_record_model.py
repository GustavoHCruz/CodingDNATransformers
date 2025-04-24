from models.base_model import MinimalModel
from models.child_dataset_model import ChildDataset
from models.parent_record_model import ParentRecord
from sqlmodel import Field, Relationship


class ChildRecord(MinimalModel, table=True):
  child_id: int = Field(foreign_key="childdataset.id", index=True)
  parent_record_id: int = Field(foreign_key="parentrecord.id", index=True)
  batch_id: int = Field(foreign_key="generationbatch.id", index=True)

  child: ChildDataset = Relationship(back_populates="assignments")
  parent_record: ParentRecord = Relationship(back_populates="child_assignments")