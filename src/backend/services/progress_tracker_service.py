from database.db import get_session
from models.progress_model import ProgressTracker
from sqlmodel import select


def create_progress(type: str) -> ProgressTracker:
  with get_session() as session:
    new_instance = ProgressTracker(type=type)
    session.add(new_instance)
    session.commit()
    session.refresh(new_instance)
    return new_instance

def set_progress(task_id: id, progress: float) -> ProgressTracker:
  with get_session() as session:
    statement = select(ProgressTracker).where(ProgressTracker.id == task_id)
    record = session.exec(statement).first()
    
    record.progress = progress
    
    session.add(record)
    session.commit()
    session.refresh(record)

    return record

def get_progress(task_id: id) -> float:
  with get_session() as session:
    statement = select(ProgressTracker).where(ProgressTracker.id == task_id)
    record = session.exec(statement).first()

    return record.progress

def finish_progress(task_id: str) -> ProgressTracker:
  with get_session() as session:
    statement = select(ProgressTracker).where(ProgressTracker.id == task_id)
    record = session.exec(statement).first()
    
    session.delete(record)
    session.commit()

    return record 