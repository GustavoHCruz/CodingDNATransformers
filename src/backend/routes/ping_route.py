from typing import Literal

from fastapi import APIRouter
from services.decorators import standard_response

router = APIRouter(prefix="/ping")

@router.get("/")
@standard_response()
def get() -> Literal["Pong"]:
  return "Pong"