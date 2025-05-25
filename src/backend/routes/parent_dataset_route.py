from dtos.parent_dataset_dto import ParentRecordGetDTO
from fastapi import APIRouter, Query
from models.base_model import ApproachEnum
from repositories.parent_dataset_repo import (get_parent_dataset,
                                              get_total_amount_by_approach)
from services.decorators import standard_response

router = APIRouter(prefix="/parent-dataset", tags=["ParentDataset"])

@router.get("/total-data")
@standard_response()
def get_count_parent_datasets(approach: ApproachEnum = Query(...)) -> int:
  response = get_total_amount_by_approach(approach)
  return response

@router.get("/{id}")
@standard_response()
def get_parent(id: int) -> ParentRecordGetDTO:
  response = get_parent_dataset(id)
  return ParentRecordGetDTO.model_validate(response)