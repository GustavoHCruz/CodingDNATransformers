from typing import Optional

from models.generation_batch_model import GenerationBatch
from services.decorators import with_session
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select


@with_session
def create_generation_batch(data: GenerationBatch, session: Optional[Session] = None) -> GenerationBatch:
  assert session is not None

  session.add(data)
  session.commit()
  session.refresh(data)
  return data

@with_session
def get_generation_batch_by_id(id: int, session: Optional[Session] = None) -> GenerationBatch | None:
  assert session is not None

  stmt = (
    select(GenerationBatch)
    .where(GenerationBatch.id == id)
    .options(selectinload(GenerationBatch.children))
  )
  record = session.exec(stmt).first()

  return record