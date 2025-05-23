from typing import Optional

from database.db import get_session
from models.child_record_model import ChildRecord
from services.decorators import handle_exceptions
from services.progress_tracker_service import post_progress
from sqlalchemy import insert
from sqlmodel import Session

from utils import chunked_generator


@handle_exceptions
def bulk_create_child_record_from_generator(data_generator, batch_size, task_id, total_records, session: Optional[Session] = None) -> int:
  own_session = False
  if session is None:
    session = get_session()
    own_session = True
  
  try:  
    counter = 0

    for batch in chunked_generator(data_generator, batch_size):
      stmt = insert(ChildRecord).values(batch)
      session.exec(stmt)
      session.commit()
      counter += batch_size

    post_progress(task_id, total_records, counter)
    
    return counter
  finally:
    if own_session:
      session.close()