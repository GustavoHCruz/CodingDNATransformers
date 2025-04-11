import json

from fastapi import HTTPException

from app.backend.schemas.configuration_schema import ConfigPatch
from app.backend.services.decorators import handle_exceptions
from app.utils.paths import read_config_file


@handle_exceptions
def get_config():
  with open(read_config_file("config.json"), "r") as file:
    data = json.load(file)
    return data

@handle_exceptions
def patch_config(data: ConfigPatch):
  with open(read_config_file("config.json"), "r") as file:
    current = json.load(file)
    update_data = data.model_dump(exclude_unset=True)

    current.update(update_data)

  with open(read_config_file("config.json"), "w") as file:
    json.dump(current, file, indent=2)
  
  return current
      