from typing import Optional

from models.progress_tracker_model import ProgressTracker, StatusEnum
from services.decorators import with_session
from sqlmodel import Session, select


@with_session
def create_progress(progress_type: str, task_name: str | None = None, info: str | None = None, session: Optional[Session] = None) -> ProgressTracker:
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
def set_progress(task_id: id, progress: float, session: Optional[Session]) -> ProgressTracker:
  stmt = select(ProgressTracker).where(ProgressTracker.id == task_id)
  record = session.exec(stmt).first()
  
  record.progress = progress
  
  session.add(record)
  session.commit()
  session.refresh(record)

  return record

@with_session
def get_progress(task_id: id, session: Optional[Session] = None) -> float:
  stmt = select(ProgressTracker).where(ProgressTracker.id == task_id)
  record = session.exec(stmt).first()

  return record.progress

@with_session
def finish_progress(task_id: str, session: Optional[Session] = None) -> ProgressTracker:
  stmt = select(ProgressTracker).where(ProgressTracker.id == task_id)
  record = session.exec(stmt).first()
  
  record.status = StatusEnum.complete
  session.add(record)
  session.commit()
  session.refresh(record)

  return record