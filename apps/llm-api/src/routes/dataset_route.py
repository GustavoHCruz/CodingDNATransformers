from fastapi import APIRouter
from schemas.datasets_schema import (CreationSettings,
                                     CreationSettingsResponse,
                                     ProcessedDatasetCreation,
                                     ProcessedDatasetCreationResponse)
from services.dataset_service import generate_processed_datasets, process_raw
from services.decorators import standard_response

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.post("/raw")
@standard_response()
def post_raw(data: CreationSettings) -> CreationSettingsResponse:
  response = process_raw(data)
  return response

@router.post("/processed")
@standard_response()
def processed_dataset_post(data: ProcessedDatasetCreation) -> list[ProcessedDatasetCreationResponse]:
  response = generate_processed_datasets(data)
  return response