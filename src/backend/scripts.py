import os

from dotenv import load_dotenv

load_dotenv()

def start_dev() -> None:
  os.system("uvicorn src.backend.main:app --reload")

def start_prod() -> None:
  os.system("uvicorn src.backend.main:app")