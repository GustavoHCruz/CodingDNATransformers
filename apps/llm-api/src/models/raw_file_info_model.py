from models.base_model import ApproachEnum, BaseModel


class RawFileInfo(BaseModel, table=True):
  file_name: str
  approach: ApproachEnum
  total_records: int
