from fastapi import APIRouter, FastAPI, HTTPException

from app.backend.api import configuration
from app.backend.exceptions.handlers import (generic_exception_handler,
                                             http_exception_handler)

app = FastAPI()

router = APIRouter(prefix="/ping")

@router.get("/")
def get():
  return "Pong"

# Global Handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Routes
app.include_router(router)
app.include_router(configuration.router)