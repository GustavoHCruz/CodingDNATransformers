from datetime import datetime

from pydantic import BaseModel


class ParentRecordGetDTO(BaseModel):
  id: int
  name: str
  origin: str
  record_count: int
  created_at: datetime

  class Config:
    from_attributes = True