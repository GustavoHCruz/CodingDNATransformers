from typing import Optional

from database.db import get_session
from models.parent_record_model import ParentRecord
from services.decorators import with_session
from services.parent_record_service import remove_duplicated
from services.progress_tracker_service import finish_progress, post_progress
from sqlmodel import Session, insert

from utils import chunked_generator


@with_session
def bulk_create_parent_record_from_generator(data_generator, batch_size, task_id, total_records, session: Optional[Session] = None) -> tuple[int, int]:
  accepted_total = 0
  raw_total = 0

  with get_session() as session:
    for batch in chunked_generator(data_generator, batch_size):
      raw_total += len(batch)
      batch = remove_duplicated(batch)
      accepted_total += len(batch)

      if batch:
        stmt = insert(ParentRecord).values(batch)  
        session.exec(stmt)
        session.commit()

      post_progress(task_id, total_records, raw_total)
  
  finish_progress(task_id)

  return accepted_total, raw_total