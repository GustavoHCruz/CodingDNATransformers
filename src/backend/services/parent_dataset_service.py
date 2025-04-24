from database.db import get_session
from models.parent_dataset_model import ParentDataset
from services.decorators import handle_exceptions
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