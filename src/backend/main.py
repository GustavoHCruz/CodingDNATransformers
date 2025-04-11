from contextlib import asynccontextmanager
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException

from backend.api import configuration
from backend.db import init_db
from backend.handlers.exception_handlers import (generic_exception_handler,
                                                 http_exception_handler)

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
  init_db()

router = APIRouter(prefix="/ping")

@router.get("/")
def get() -> Literal['Pong']:
  return "Pong"

# Global Handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Routes
app.include_router(router)
app.include_router(configuration.router)
