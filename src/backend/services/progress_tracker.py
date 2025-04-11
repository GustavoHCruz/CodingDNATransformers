from threading import Lock
from typing import Dict

progress_data = Dict[str, int] = {}
lock = Lock()

def set_progress(task_id: str, percent: int):
  with lock:
    progress_data[task_id] = percent

def get_progress(task_id: str) -> int:
  with lock:
    return progress_data.get(task_id, 0)