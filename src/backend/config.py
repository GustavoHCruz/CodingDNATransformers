import yaml


class Config:
  def __init__(self, path: str = "config.yaml") -> None:
    self._config = self._load_config(path)
  
  def _load_config(self, path: str) -> dict:
    with open(path, "r") as f:
      return yaml.safe_load(f)
    
  def get(self, *keys, default=None):
    value = self._config
    for key in keys:
      if key in value:
        value = value[key]
      else:
        return default
    
    return value
  
config = Config()