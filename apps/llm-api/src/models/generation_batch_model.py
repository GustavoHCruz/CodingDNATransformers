from typing import List

from sqlmodel import Relationship

from models.base_model import BaseModel


class GenerationBatch(BaseModel, table=True):
  name: str
  
  children: List["ChildDataset"] = Relationship(back_populates="batch")