from enum import Enum


class ModelEnum(str, Enum):
  GPT = "GPT"
  BERT = "BERT"
  DNABERT = "DNABERT"

class ApproachEnum(str, Enum):
  EXINCLASSIFIER = "EXINCLASSIFIER"
  TRIPLETCLASSIFIER = "TRIPLETCLASSIFIER"
  DNATRANSLATOR = "DNATRANSLATOR"
