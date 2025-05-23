import select
from typing import Optional

from database.db import get_session
from models.child_dataset_model import ChildDataset
from services.decorators import handle_exceptions
from sqlalchemy import func
from sqlmodel import Session


@handle_exceptions
def create_child_dataset(data: ChildDataset, session: Optional[Session] = None) -> ChildDataset:
  own_session = False
  if session is None:
    session = get_session()
    own_session = True

  try:  
    session.add(data)
    session.commit()
    session.refresh(data)
    return data
  finally:
    if own_session:
      session.close()

@handle_exceptions
def get_batch_amount(batch_id: int, parent_id: int) -> int:
  with get_session() as session:
    statement = (
      select(func.sum(ChildDataset.record_count))
      .where(ChildDataset.batch_id == batch_id)
      .where(ChildDataset.parent_id == parent_id)
    )

    result = session.exec(statement).one_or_none()

    return result[0] or 0