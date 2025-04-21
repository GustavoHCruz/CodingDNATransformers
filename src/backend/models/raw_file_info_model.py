from models.base_model import BaseModel


class RawFileInfo(BaseModel, table=True):
  file_name: str
  total_records: int
