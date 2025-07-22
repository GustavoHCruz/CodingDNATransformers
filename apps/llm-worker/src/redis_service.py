from enum import Enum
from typing import Union

import redis
from dotenv import dotenv_values, load_dotenv

load_dotenv()
config = dotenv_values(".env")

class ProcessingStatus(str, Enum):
  IN_PROGRESS = "in_progress"
  DONE = "done"

class CreateField(str, Enum):
  STATUS = "status"

class TrainField(str, Enum):
  STATUS = "status"
  GPU_AMOUNT = "gpus_amount"
  TOTAL_STEPS = "total_steps"
  STEP = "step"
  LOSS = "loss"
  LR = "lr"

class EvalField(str, Enum):
  STATUS = "status"
  ACCURACY = "accuracy"
  LOSS = "loss"

class PredictField(str, Enum):
  STATUS = "status"
  OUTPUT = "output"

class RedisService:
  def __init__(self):
    host = config.get("REDIS_HOST")
    port = config.get("REDIS_PORT", 6379)

    if not host:
      raise ValueError("REDIS_HOST environment key is missing")

    if not port:
      raise ValueError("REDIS_PORT environment key is missing")

    self.redis = redis.Redis(
      host=host,
      port=int(port),
      db=0,
      decode_responses=True
    )
    self.default_ex = 86400
  
  def set_create_info(self, uuid: str, field: CreateField, value: ProcessingStatus) -> None:
    self.redis.set(name=f"create:{uuid}:{field.value}", value=value, ex=self.default_ex)

  def set_train_info(self, uuid: str, field: TrainField, value: Union[str, int, float, ProcessingStatus]) -> None:
    self.redis.set(name=f"train:{uuid}:{field.value}", value=value, ex=self.default_ex)

  def set_eval_info(self, uuid: str, field: EvalField, value: Union[str, int, float, ProcessingStatus]) -> None:
    self.redis.set(name=f"eval:{uuid}:{field.value}", value=value, ex=self.default_ex)

  def set_predict_info(self, uuid: str, field: PredictField, value: Union[str, int, float, ProcessingStatus]) -> None:
    self.redis.set(name=f"predict:{uuid}:{field.value}", value=value, ex=self.default_ex)

  def get_key(self, key: str) -> str:
    return str(self.redis.get(name=key))

  def delete_key(self, key: str):
    return self.redis.delete(key)
  
  def ttl(self, key: str):
    return self.redis.ttl(key)

redis_service = RedisService()