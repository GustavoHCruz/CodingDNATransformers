from typing import cast

from config import Config
from llms.exin_classifier.bert import ExInSeqsBERT
from llms.exin_classifier.gpt import ExInSeqsGPT
from models.base_model import ApproachEnum
from schemas.fine_tuning_schema import ModelCreateRequest, ModelTypeEnum

supported_berts = ["bert-base-uncased"]
supported_gpts = []
supported_t5s = []

def create_model(data: ModelCreateRequest):
  configs = Config()

  models_dir = configs.paths.models
  logs_dir = configs.paths.model_logs_dir

  if data.approach == ApproachEnum.exin_classifier:
    if data.model_type == ModelTypeEnum.gpt:
      model = ExInSeqsGPT(
        
      )
    elif data.model_type == ModelTypeEnum.bert:
      model = ExInSeqsBERT(
        checkpoint=data.model_name,
        alias=data.name,
        models_dir=models_dir,
        logs_dir=logs_dir,
        seed=data.seed
      )
  elif data.approach == ApproachEnum.exin_translator:
    model = Tran