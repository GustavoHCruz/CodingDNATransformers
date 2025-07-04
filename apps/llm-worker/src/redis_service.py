import redis
from dotenv import dotenv_values, load_dotenv

load_dotenv()
config = dotenv_values(".env")

class RedisService:
    def __init__(self, host=config.get("REDIS_HOST"), port=None, db=0, decode_responses=True):
      self.redis = redis.Redis(
        host=host,
        port=int(port or config.get("REDIS_PORT", 6379)),
        db=db,
        decode_responses=decode_responses
      )

    def set_key(self, key: str, value: str, ex: int = None):
      self.redis.set(name=key, value=value, ex=ex)

    def get_key(self, key: str) -> str:
      return self.redis.get(name=key)

    def delete_key(self, key: str):
      return self.redis.delete(key)
    
    def ttl(self, key: str):
      return self.redis.ttl(key)

redis_service = redis.Redis(config.get("REDIS_HOST"), port=config["REDIS_PORT"])