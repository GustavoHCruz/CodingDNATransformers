from typing import Literal

from fastapi import APIRouter
from services.decorators import handle_exceptions

router = APIRouter(prefix="/ping")

@handle_exceptions
@router.get("/")
def get() -> Literal["Pong"]:
  return "Pong"