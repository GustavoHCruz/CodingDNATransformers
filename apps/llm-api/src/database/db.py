from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session

from config import config
from models.base_model import BaseModel
from models.child_dataset_model import ChildDataset
from models.child_record_model import ChildRecord
from models.generation_batch_model import GenerationBatch
from models.parent_dataset_model import ParentDataset
from models.parent_record_model import ParentRecord
from models.progress_tracker_model import ProgressTracker
from models.raw_file_info_model import RawFileInfo

DATABASE_URL = config.database.url

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record) -> None:
  cursor = dbapi_connection.cursor()
  cursor.execute("PRAGMA synchronous = OFF")
  cursor.execute("PRAGMA journal_mode = MEMORY")
  cursor.close()

SessionLocal = sessionmaker(bind=engine, class_=Session, autocommit=False, autoflush=False, expire_on_commit=False)

models = [
  ChildDataset,
  ChildRecord,
  GenerationBatch,
  ParentDataset,
  ParentRecord,
  ProgressTracker,
  RawFileInfo,
]

for model in models:
  event.listen(model, "before_update", BaseModel._update_timestamp)

def init_db() -> None:
  for model in models:
    model.metadata.create_all(bind=engine)

@contextmanager
def get_session() -> Generator[Session, None, None]:
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()
