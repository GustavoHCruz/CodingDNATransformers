from typing import Optional

from models.child_dataset_model import ChildDataset
from services.decorators import with_session
from sqlalchemy import func
from sqlmodel import Session, select


@with_session
def create_child_dataset(data: ChildDataset, session: Optional[Session] = None) -> ChildDataset:
  assert session is not None

  session.add(data)
  session.commit()
  session.refresh(data)
  return data

@with_session
def get_batch_amount(batch_id: int, session: Optional[Session] = None) -> int:
  assert session is not None

  stmt = select(func.sum(ChildDataset.record_count)).where(
    ChildDataset.batch_id == batch_id
  )
  result = session.exec(stmt).one_or_none()

  total = result

  return total or 0