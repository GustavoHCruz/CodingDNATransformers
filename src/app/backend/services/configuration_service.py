import json

from fastapi import HTTPException

from app.backend.schemas.configuration_schema import ConfigPatch
from app.utils.paths import config_file


def get_config():
  try:
    with open(config_file("config.json"), "r") as file:
      data = json.load(file)
      return data
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

def patch_config(data: ConfigPatch):
  try:
    with open(config_file("config.json"), "r") as file:
      current = json.load(file)
      update_data = data.model_dump(exclude_unset=True)

      current.update(update_data)

    with open(config_file("config.json"), "w") as file:
      json.dump(current, file, indent=2)
    
    return current
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
      