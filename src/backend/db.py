from models.progress import Progress
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session

DATABASE_URL = "sqlite:///./splicingsitestransformers.db"

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def init_db() -> None:
  Progress.metadata.create_all(bind=engine)

def get_session() -> Session:
  return Session(engine)