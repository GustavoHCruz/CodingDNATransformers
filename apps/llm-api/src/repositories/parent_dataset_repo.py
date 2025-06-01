from typing import Optional

from fastapi import HTTPException, status
from models.base_model import ApproachEnum
from models.parent_dataset_model import OriginEnum, ParentDataset
from services.decorators import with_session
from sqlalchemy import func
from sqlmodel import Session, select


@with_session
def create_parent_dataset(data: ParentDataset, session: Optional[Session] = None) -> ParentDataset:
  assert session is not None

  session.add(data)
  session.commit()
  session.refresh(data)
  return data

@with_session
def update_parent_dataset_record_counter(id: int, counter: int, session: Optional[Session] = None) -> ParentDataset:
  assert session is not None

  stmt = select(ParentDataset).where(ParentDataset.id == id)
  record = session.exec(stmt).first()

  if not record:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Parent dataset with ID {id} not found"
    )

  record.record_count = counter
  session.add(record)
  session.commit()
  session.refresh(record)

  return record

@with_session
def get_parent_dataset(id: int, session: Optional[Session] = None) -> ParentDataset:
  assert session is not None

  stmt = select(ParentDataset).where(ParentDataset.id == id)

  result = session.exec(stmt).first()

  if result is None:
    raise ValueError("ParentDataset Not Found")

  return result

@with_session
def get_total_amount_by_approach(approach: ApproachEnum, session: Optional[Session] = None) -> int:
  assert session is not None

  stmt = (
    select(func.sum(ParentDataset.record_count))
    .where(ParentDataset.approach == approach)
  )

  result = session.exec(stmt).one_or_none()

  return result or 0

@with_session
def get_total_amount_by_approach_and_origin(approach: ApproachEnum, origin:OriginEnum, session: Optional[Session] = None) -> int:
  assert session is not None
  
  stmt = (
    select(func.sum(ParentDataset.record_count))
    .where(ParentDataset.approach == approach)
    .where(ParentDataset.origin == origin)
  )

  result = session.exec(stmt).one_or_none()

  return result[0] if result else 0

@with_session
def get_parents_datasets_ids_by_approach(approach: ApproachEnum, session: Optional[Session] = None) -> list[int | None]:
  assert session is not None

  stmt = select(ParentDataset.id).where(ParentDataset.approach == approach)

  result = session.exec(stmt).all()

  return list(result)