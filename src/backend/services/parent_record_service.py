from typing import List

from database.db import get_session
from models.parent_record_model import ParentRecord
from services.decorators import handle_exceptions
from sqlmodel import insert

seen_global = set()

def remove_duplicated(instances: List[ParentRecord]) -> List[ParentRecord]:
  global seen_global
  unique = []

  for instance in instances:
    key = (
      instance["sequence"],
      instance["target"],
      instance["flank_before"],
      instance["flank_after"],
      instance["organism"],
      instance["gene"])

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
def bulk_create_parent_record_from_generator(data_generator, batch_size: int = 3000) -> int:
  total = 0

  with get_session() as session:
    for batch in chunked_generator(data_generator, batch_size):
      batch = remove_duplicated(batch)
      total += len(batch)
    
      stmt = insert(ParentRecord).values(batch)  

      session.exec(stmt)
      session.commit()

  return total