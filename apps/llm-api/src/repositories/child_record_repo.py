from typing import Callable, Optional

from sqlalchemy import insert
from sqlmodel import Session

from models.child_record_model import ChildRecord
from repositories.progress_tracker_repo import finish_progress, set_progress
from services.decorators import with_session
from services.progress_tracker_service import post_progress
from utils import chunked_generator


@with_session
def bulk_create_child_records_from_generator(data_generator: Callable, task_id: int, total_records: int, batch_size: Optional[int] = 250, session: Optional[Session] = None) -> int:
  counter = 0

  for batch in chunked_generator(data_generator, batch_size):
    stmt = insert(ChildRecord).values(batch)
    session.exec(stmt)
    session.commit()
    counter += batch_size

    post_progress(task_id, total_records, counter)

  finish_progress(task_id)
  
  return counter

@with_session
def bulk_create_child_records(data, session: Optional[Session] = None) -> bool:
  stmt = insert(ChildRecord).values(data)
  session.exec(stmt)
  session.commit()

  return True