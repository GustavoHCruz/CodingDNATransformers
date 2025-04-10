from fastapi import APIRouter

from app.backend.models.base import BaseResponse
from app.backend.models.configuration_model import ConfigModel
from app.backend.schemas.configuration_schema import ConfigPatch
from app.backend.services.configuration_service import get_config, patch_config

router = APIRouter(prefix="/configuration", tags=["Configuration"])

@router.get("/")
def get():
  data = get_config()
  data = ConfigModel(**data)
  return BaseResponse(status="success", message="Configuration loaded", data=data)

@router.patch("/")
def patch(data: ConfigPatch):
  response = patch_config(data)
  return BaseResponse(status="success", message="Configuration File Updated", data=response)