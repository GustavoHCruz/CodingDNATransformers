from enum import Enum

from models.base_model import ApproachEnum
from pydantic import BaseModel, field_validator


class ModelTypeEnum(str, Enum):
  gpt = "gpt"
  bert = "bert"
  dnabert = "dnabert"
  t5 = "t5"

GPT_MODELS = {
  "gpt2",
  "gpt2-medium",
  "gpt2-large",
  "gpt2-xl",
  "EleutherAI/gpt-neo-125M",
  "EleutherAI/gpt-neo-1.3B",
  "bigscience/bloom-560m"
}

BERT_MODELS = {
  "bert-base-uncased",
}

DNABERT_MODELS = {
  "dnabert"
}

T5_MODELS = {
  "t5-small",
  "t5-base",
  "t5-large",
  "flan-t5-small",
  "flan-t5-base",
  "flan-t5-large"
}

class ModelCreateRequest(BaseModel):
  approach: ApproachEnum
  model_type: ModelTypeEnum
  model_name: str
  name: str
  seed: str

  @field_validator("model_name")
  def validate_model_name(cls, model_name, values):
    approach = values.get("approach")
    model_type = values.get("model_type")

    if approach == ApproachEnum.exin_classifier:
      if model_type == ModelTypeEnum.gpt:
        if model_name not in GPT_MODELS:
          raise ValueError(f"Model '{model_name}' is not a valid GPT model.")
      elif model_type == ModelTypeEnum.bert:
        if model_name not in BERT_MODELS:
          raise ValueError(f"Model '{model_name}' is not a valid BERT model.")
      elif model_type == ModelTypeEnum.dnabert:
        if model_name not in DNABERT_MODELS:
          raise ValueError(f"Model '{model_name}' is not a valid DNABERT model.")
      else:
        raise ValueError(f"Unsupported model type for approach {approach.value}.")
    elif approach == ApproachEnum.exin_translator:
      if model_type == ModelTypeEnum.gpt:
        if model_name not in GPT_MODELS:
          raise ValueError(f"Model '{model_name}' is not a valid GPT model.")
      if model_type == ModelTypeEnum.t5:
        if model_name not in T5_MODELS:
          raise ValueError(f"Model '{model_name}' is not a valid T5 model.")
      else:
        raise ValueError(f"Unsupported model type for approach {approach.value}.")
    elif approach == ApproachEnum.protein_translator:
      if model_type == ModelTypeEnum.gpt:
        if model_name not in GPT_MODELS:
          raise ValueError(f"Model '{model_name}' is not a valid GPT model.")
      elif model_type == ModelTypeEnum.t5:
        if model_name not in T5_MODELS:
          raise ValueError(f"Model '{model_name}' is not a valid T5 model.")
      else:
        raise ValueError(f"Unsupported model type for approach {approach.value}.")
    elif approach == ApproachEnum.sliding_window_extraction:
      if model_type == ModelTypeEnum.bert:
        if model_name not in BERT_MODELS:
          raise ValueError(f"Model '{model_name}' is not a valid BERT model.")
      elif model_type == ModelTypeEnum.dnabert:
        if model_name not in DNABERT_MODELS:
          raise ValueError(f"Model '{model_name}' is not a valid DNABERT model.")
      else:
        raise ValueError(f"Unsupported model type for approach {approach.value}.")
    else:
      raise ValueError(f"Unsupported approach: {approach}")

    return model_name
