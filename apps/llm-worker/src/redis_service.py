from typing import Literal

import redis
from dotenv import dotenv_values, load_dotenv

load_dotenv()
config = dotenv_values(".env")

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
  
  def set_create_status(self, uuid: str, status: Literal["IN_PROGRESS", "DONE"]) -> None:
    self.redis.set(name=f"create:{uuid}:status", value=status, ex=self.default_ex)

  def set_train_gpu_amount(self, uuid: str, gpu_amount: int) -> None:
    self.redis.set(name=f"train:{uuid}:gpus_amount", value=gpu_amount, ex=self.default_ex)

  def set_train_total_steps(self, uuid: str, total_steps: int) -> None:
    self.redis.set(name=f"train:{uuid}:total_steps", value=total_steps, ex=self.default_ex)

  def set_train_step(self, uuid: str, step: int) -> None:
    self.redis.set(name=f"train:{uuid}:step", value=step, ex=self.default_ex)
  
  def set_train_loss(self, uuid: str, loss: float) -> None:
    self.redis.set(name=f"train:{uuid}:loss", value=loss, ex=self.default_ex)
  
  def set_train_lr(self, uuid: str, lr: float) -> None:
    self.redis.set(name=f"train:{uuid}:lr", value=lr, ex=self.default_ex)
  
  def set_train_status(self, uuid: str, status: Literal["IN_PROGRESS", "DONE"]):
    self.redis.set(name=f"train:{uuid}:status", value=status, ex=self.default_ex)

  def get_key(self, key: str) -> str:
    return str(self.redis.get(name=key))

  def delete_key(self, key: str):
    return self.redis.delete(key)
  
  def ttl(self, key: str):
    return self.redis.ttl(key)

redis_service = RedisService()