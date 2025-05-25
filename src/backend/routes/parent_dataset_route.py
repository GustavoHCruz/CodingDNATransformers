from fastapi.routing import APIRoute
from repositories.parent_dataset_repo import get_parent_dataset
from schemas.base_response import BaseResponse
from services.decorators import handle_exceptions

router = APIRoute(prefix="/parent_dataset", tags=["ParentDataset"])

@router.get("/{id}")
@handle_exceptions
def get_parent(id: int) -> BaseResponse:
  response = get_parent_dataset(id)
  return BaseResponse(status="success", data=response)