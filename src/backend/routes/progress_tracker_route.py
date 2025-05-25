from dtos.progress_tracker_dto import ProgressTrackerGetDTO
from fastapi import APIRouter
from models.progress_tracker_model import ProgressTracker
from repositories.progress_tracker_repo import get_progress
from services.decorators import standard_response

router = APIRouter(prefix="/progress-tracker")

@router.get("/{progress_tracker_id}")
@standard_response()
def get(progress_tracker_id: int) -> ProgressTrackerGetDTO:
  response = get_progress(progress_tracker_id)

  return ProgressTrackerGetDTO.model_validate(response)