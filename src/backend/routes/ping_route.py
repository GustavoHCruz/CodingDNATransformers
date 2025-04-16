from typing import Literal

from fastapi import APIRouter

router = APIRouter(prefix="/ping")

@router.get("/")
def get() -> Literal["Pong"]:
  return "Pong"