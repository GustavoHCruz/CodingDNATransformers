from typing import Literal

from pydantic import BaseModel


class DatabaseConfig(BaseModel):
	url: str
	batch_size: int


class PathsConfig(BaseModel):
	raw_data: str
	models: str
	model_logs_dir: str


class LoggingConfig(BaseModel):
	level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
	file: str


class AppConfig(BaseModel):
	database: DatabaseConfig
	paths: PathsConfig
	logging: LoggingConfig
