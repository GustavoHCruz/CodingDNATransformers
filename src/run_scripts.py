import os
import subprocess


def backend_dev() -> None:
  os.chdir("src")
  os.chdir("backend")
  subprocess.run(["uvicorn", "app:app", "--reload"])

def backend() -> None:
  os.chdir("src")
  os.chdir("backend")
  subprocess.run(["python", "main.py"])

def frontend_dev() -> None:
  os.chdir("src")
  os.chdir("frontend")
  subprocess.run()

def frontend() -> None:
  os.chdir("scr")
  os.chdir("frontend")
  subprocess.run()