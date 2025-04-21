from fastapi.routing import APIRoute
from services.progress_tracker_service import get_progress

from src.backend.schemas.base_response import BaseResponse

router = APIRoute(prefix="/progress_tracker")

@router.get("/")
def get(task_id: int) -> BaseResponse:
  return BaseResponse(get_progress(task_id))