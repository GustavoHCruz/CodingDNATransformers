from tqdm import tqdm
from typing import Optional, TypedDict

class TqdmArgs(TypedDict, total=False):
  bar_format: Optional[str]
  desc: Optional[str]
  total: Optional[str]
  leave: Optional[str]


class ProgressTracker():  
  def __init__(self, from_terminal=True, info:TqdmArgs={}) -> None:
    if from_terminal:
      self.progress_bar = tqdm(**info)
    else:
      


def update_progress_tracker():

def end_progress_tracker():
