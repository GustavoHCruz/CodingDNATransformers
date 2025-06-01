from contextlib import asynccontextmanager

from database.db import init_db
from fastapi import FastAPI


async def setup_all() -> None:
  init_db()

@asynccontextmanager
async def lifespan(_: FastAPI):
  init_db()
  yield