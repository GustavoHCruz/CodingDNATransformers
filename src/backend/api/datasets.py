from fastapi import APIRouter

from app.backend.models.base import BaseResponse
from app.backend.schemas.datasets_schema import DatasetsRaw
from app.backend.services.datasets_service import process_raw

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.patch("/")
def patch(data: DatasetsRaw):
  response = patch_config(data)
  return BaseResponse(status="success", message="Configuration File Updated", data=response)