import yaml
from schemas.app_config_schema import (AppConfig, DatabaseConfig,
                                       LoggingConfig, PathsConfig)


class Config:
  def __init__(self, path: str = "config.yaml") -> None:
    self._config = self._load_config(path)
  
  def _load_config(self, path: str) -> AppConfig:
    with open(path, "r") as f:
      data = yaml.safe_load(f)
    return AppConfig.model_validate(data)
    
  @property
  def database(self) -> DatabaseConfig:
    return self._config.database

  @property
  def paths(self) -> PathsConfig:
    return self._config.paths
  
  @property
  def logging(self) -> LoggingConfig:
    return self._config.logging

config = Config()