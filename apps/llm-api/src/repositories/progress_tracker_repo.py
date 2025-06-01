from typing import Optional

from fastapi import HTTPException, status
from models.progress_tracker_model import (ProgressTracker, ProgressTypeEnum,
                                           StatusEnum)
from services.decorators import with_session
from sqlmodel import Session, select


@with_session
def create_progress(progress_type: ProgressTypeEnum, task_name: str | None = None, info: str | None = None, session: Optional[Session] = None) -> ProgressTracker:
  assert session is not None

  new_instance = ProgressTracker(
    progress_type=progress_type,
    task_name=task_name,
    info=info
  )
  session.add(new_instance)
  session.commit()
  session.refresh(new_instance)
  return new_instance

@with_session
def set_progress(task_id: int, progress: float, session: Optional[Session]) -> ProgressTracker:
  assert session is not None

  stmt = select(ProgressTracker).where(ProgressTracker.id == task_id)
  record = session.exec(stmt).first()
  if not record:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Progress tracker with ID {task_id} not found"
    )
  record.progress = progress
  
  session.add(record)
  session.commit()
  session.refresh(record)

  return record

@with_session
def get_progress(task_id: int, session: Optional[Session] = None) -> ProgressTracker | None:
  if not session:
    raise ValueError("Could not instantiate session") 

  stmt = select(ProgressTracker).where(ProgressTracker.id == task_id)
  record = session.exec(stmt).first()

  return record

@with_session
def finish_progress(task_id: int, session: Optional[Session] = None) -> ProgressTracker:
  assert session is not None
  
  stmt = select(ProgressTracker).where(ProgressTracker.id == task_id)
  record = session.exec(stmt).first()
  if not record:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Progress tracker with ID {task_id} not found"
    )
  
  record.status = StatusEnum.complete
  session.add(record)
  session.commit()
  session.refresh(record)

  return record