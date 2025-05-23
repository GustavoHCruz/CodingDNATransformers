from database.db import get_session
from models.base_model import ApproachEnum
from models.parent_dataset_model import OriginEnum, ParentDataset
from services.decorators import handle_exceptions
from sqlalchemy import func
from sqlmodel import select


@handle_exceptions
def create_parent_dataset(data: ParentDataset) -> ParentDataset:
  with get_session() as session:
    session.add(data)
    session.commit()
    session.refresh(data)
    return data

@handle_exceptions
def update_parent_dataset_record_counter(id: int, counter: int) -> ParentDataset:
  with get_session() as session:
    statement = select(ParentDataset).where(ParentDataset.id == id)
    record = session.exec(statement).first()

    record.record_count = counter
    session.add(record)
    session.commit()
    session.refresh(record)

    return record

@handle_exceptions
def get_parent_dataset(id: int) -> ParentDataset:
  with get_session() as session:
    stmt = select(ParentDataset).where(ParentDataset.id == id)
  
    result = session.exec(stmt).first()

    if result is None:
      raise ValueError("ParentDataset Not Found")

    return result

@handle_exceptions
def get_total_amount(approach: ApproachEnum, origin:OriginEnum) -> int:
  with get_session() as session:
    statement = (
      select(func.sum(ParentDataset.record_count))
      .where(ParentDataset.approach == approach)
      .where(ParentDataset.origin == origin)
    )

    result = session.exec(statement).one_or_none()

    return result[0] or 0