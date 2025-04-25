from database.db import get_session
from models.progress_tracker_model import ProgressTracker, StatusEnum
from services.decorators import handle_exceptions
from sqlmodel import select


@handle_exceptions
def create_progress(progress_type: str, task_name: str | None = None, info: str | None = None) -> ProgressTracker:
  with get_session() as session:
    new_instance = ProgressTracker(
      progress_type=progress_type,
      task_name=task_name,
      info=info
    )
    session.add(new_instance)
    session.commit()
    session.refresh(new_instance)
    return new_instance

@handle_exceptions
def set_progress(task_id: id, progress: float) -> ProgressTracker:
  with get_session() as session:
    statement = select(ProgressTracker).where(ProgressTracker.id == task_id)
    record = session.exec(statement).first()
    
    record.progress = progress
    
    session.add(record)
    session.commit()
    session.refresh(record)

    return record

@handle_exceptions
def get_progress(task_id: id) -> float:
  with get_session() as session:
    statement = select(ProgressTracker).where(ProgressTracker.id == task_id)
    record = session.exec(statement).first()

    return record.progress

@handle_exceptions
def finish_progress(task_id: str) -> ProgressTracker:
  with get_session() as session:
    statement = select(ProgressTracker).where(ProgressTracker.id == task_id)
    record = session.exec(statement).first()
    
    record.status = StatusEnum.complete
    session.add(record)
    session.commit()
    session.refresh(record)

    return record 

def post_progress(task_id: int, total_records: int | None, counter: int) -> None:
	progress = counter
	if total_records:
		progress = (counter * 100) / total_records

	set_progress(task_id, progress)