from fastapi import APIRouter

from src.funcs.config_reading import read_config_file

router = APIRouter(prefix="/configuration", tags=["Configuration"])

@router.get("/")
def get():
  response = read_config_file()
  return {"response": response}