from fastapi import APIRouter
from models.generation_batch_model import GenerationBatch
from schemas.base_response import BaseResponse
from schemas.datasets_schema import CreationSettings
from schemas.processed_datasets_schema import ProcessedDatasetCreation
from services.dataset_service import generate_processed_datasets, process_raw
from services.decorators import handle_exceptions

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.post("/raw")
@handle_exceptions
def post_raw(data: CreationSettings) -> BaseResponse:
  response = process_raw(data)
  return BaseResponse(status="success", message="Created", data=response)

@handle_exceptions
@router.post("/processed")
def processed_dataset_post(data: ProcessedDatasetCreation) -> GenerationBatch:
  response = generate_processed_datasets(data)
  return response