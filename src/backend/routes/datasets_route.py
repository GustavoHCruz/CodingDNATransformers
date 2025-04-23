from fastapi import APIRouter
from schemas.base_response import BaseResponse
from schemas.datasets_schema import CreationSettings
from services.datasets_service import process_raw

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.post("/")
def post(data: CreationSettings) -> BaseResponse:
  response = process_raw(data)
  return BaseResponse(status="success", message="Created", data=response)