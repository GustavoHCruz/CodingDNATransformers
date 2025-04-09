import os

from dotenv import load_dotenv

load_dotenv()

def start_dev():
  os.system("uvicorn src.app.backend.main:app --reload")

def start_prod():
  os.system("uvicorn src.app.backend.main:app")