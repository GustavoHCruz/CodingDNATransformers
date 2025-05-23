from fastapi import APIRouter
from schemas.base_response import BaseResponse
from schemas.datasets_schema import CreationSettings
from services.dataset_service import process_raw
from services.parent_dataset_service import get_parent_dataset

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.post("/raw")
def post_raw(data: CreationSettings) -> BaseResponse:
  response = process_raw(data)
  return BaseResponse(status="success", message="Created", data=response)

@router.get("/parent/{id}")
def get_parent(id: int) -> BaseResponse:
  response = get_parent_dataset(id)
  return BaseResponse(status="success", data=response)