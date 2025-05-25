from typing import Optional

from models.child_record_model import ChildRecord
from services.decorators import with_session
from services.progress_tracker_service import post_progress
from sqlalchemy import insert
from sqlmodel import Session

from utils import chunked_generator


@with_session
def bulk_create_child_record_from_generator(data_generator, batch_size, task_id, total_records, session: Optional[Session] = None) -> int:
  counter = 0

  for batch in chunked_generator(data_generator, batch_size):
    stmt = insert(ChildRecord).values(batch)
    session.exec(stmt)
    session.commit()
    counter += batch_size

  post_progress(task_id, total_records, counter)
  
  return counter