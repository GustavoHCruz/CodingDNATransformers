import os
import subprocess


def backend_dev() -> None:
  os.chdir("src")
  subprocess.run(["uvicorn", "app:app", "--reload"])

def backend() -> None:
  os.chdir("src")
  subprocess.run(["python", "main.py"])