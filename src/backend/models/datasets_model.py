from enum import Enum

from models.base_model import BaseModel


class ApproachEnum(str, Enum):
  exin_classifier = "exin_classifier"
  exin_translator = "exin_translator"
  protein_translator = "protein_translator"
  sliding_window_extraction = "sliding_window_extraction"

class Datasets(BaseModel, table=True):
  approach: ApproachEnum
  name: str
  target_id: int