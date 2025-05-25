import select
from typing import Optional

from models.child_dataset_model import ChildDataset
from services.decorators import with_session
from sqlalchemy import func
from sqlmodel import Session


@with_session
def create_child_dataset(data: ChildDataset, session: Optional[Session] = None) -> ChildDataset:
  session.add(data)
  session.commit()
  session.refresh(data)
  return data

@with_session
def get_batch_amount(batch_id: int, parent_id: int, session: Optional[Session] = None) -> int:
  stmt = (
    select(func.sum(ChildDataset.record_count))
    .where(ChildDataset.batch_id == batch_id)
    .where(ChildDataset.parent_id == parent_id)
  )

  result = session.exec(stmt).one_or_none()

  return result[0] or 0