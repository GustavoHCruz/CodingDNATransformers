from typing import Optional

from models.base_model import ApproachEnum
from models.parent_dataset_model import OriginEnum, ParentDataset
from services.decorators import with_session
from sqlalchemy import func
from sqlmodel import Session, select


@with_session
def create_parent_dataset(data: ParentDataset, session: Optional[Session] = None) -> ParentDataset:
  session.add(data)
  session.commit()
  session.refresh(data)
  return data

@with_session
def update_parent_dataset_record_counter(id: int, counter: int, session: Optional[Session]) -> ParentDataset:
  stmt = select(ParentDataset).where(ParentDataset.id == id)
  record = session.exec(stmt).first()

  record.record_count = counter
  session.add(record)
  session.commit()
  session.refresh(record)

  return record

@with_session
def get_parent_dataset(id: int, session: Optional[Session] = None) -> ParentDataset:
  stmt = select(ParentDataset).where(ParentDataset.id == id)

  result = session.exec(stmt).first()

  if result is None:
    raise ValueError("ParentDataset Not Found")

  return result

@with_session
def get_total_amount(approach: ApproachEnum, origin:OriginEnum, session: Optional[Session] = None) -> int:
  stmt = (
    select(func.sum(ParentDataset.record_count))
    .where(ParentDataset.approach == approach)
    .where(ParentDataset.origin == origin)
  )

  result = session.exec(stmt).one_or_none()

  return result[0] or 0