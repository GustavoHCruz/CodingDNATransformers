from typing import Callable, Optional

from models.base_model import ApproachEnum
from models.parent_record_model import ParentRecord
from repositories.parent_dataset_repo import \
    get_parents_datasets_ids_by_approach
from repositories.progress_tracker_repo import finish_progress
from services.decorators import with_session
from services.parent_record_service import remove_duplicated
from services.progress_tracker_service import post_progress
from sqlmodel import Session, insert, select

from utils import chunked_generator


@with_session
def bulk_create_parent_record_from_generator(data_generator: Callable, batch_size: int, task_id: int, total_records: int, session: Optional[Session] = None) -> tuple[int, int]:
  if not session:
    raise ValueError("Could not instantiate session") 
  
  accepted_total = 0
  raw_total = 0

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

@with_session
def get_parents_records_ids_by_approach(approach: ApproachEnum, total_amount: int, session: Optional[Session] = None) -> list[int]:
  parents_ids = get_parents_datasets_ids_by_approach(approach=approach, session=session)
  
  stmt = select(ParentRecord.id).where(ParentRecord.parent_id.in_(parents_ids))

  result = session.exec(stmt).all()

  return result[:total_amount]