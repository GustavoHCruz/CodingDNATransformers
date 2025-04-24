from typing import List

from models.base_model import BaseModel
from sqlmodel import Relationship


class GenerationBatch(BaseModel, table=True):
  name: str
  
  children: List["ChildDataset"] = Relationship(back_populates="batch")