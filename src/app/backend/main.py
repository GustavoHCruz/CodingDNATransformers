from fastapi import APIRouter, FastAPI

from app.backend.api import configuration

app = FastAPI()

router = APIRouter(prefix="/ping")

@router.get("/")
def get():
  return "Pong"

app.include_router(router)
app.include_router(configuration.router)
