from typing import List

from database.db import get_session
from models.parent_record_model import ParentRecord
from services.decorators import handle_exceptions
from services.progress_tracker_service import finish_progress, post_progress
from sqlmodel import insert

seen_global = set()

def remove_duplicated(instances: List[ParentRecord]) -> List[ParentRecord]:
  global seen_global
  unique = []

  key_fields = ["sequence", "target", "flank_before", "flank_after", "organism", "gene"]

  for instance in instances:
    key = tuple(instance.get(field, "") for field in key_fields)

    if key not in seen_global:
      seen_global.add(key)
      unique.append(instance)
  
  return unique

def chunked_generator(generator, size):
  batch = []
  for item in generator:
    batch.append(item)
    if len(batch) == size:
      yield batch
      batch = []
  if batch:
    yield batch

@handle_exceptions
def bulk_create_parent_record_from_generator(data_generator, batch_size, task_id, total_records) -> int:
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