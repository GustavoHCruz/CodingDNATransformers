from models.base_model import BaseModel
from models.datasets_model import Datasets
from models.exin_classifier_model import ExInClassifier
from models.exin_translator_model import ExInTranslator
from models.progress_tracker_model import ProgressTracker
from models.protein_translator_model import ProteinTranslator
from models.raw_file_info_model import RawFileInfo
from models.sliding_window_tagger_model import SlidingWindowTagger
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session

DATABASE_URL = "sqlite:///./database/splicingsitestransformers.db"

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

models = [
  Datasets, 
  ExInClassifier, 
  ExInTranslator, 
  ProgressTracker, 
  ProteinTranslator, 
  RawFileInfo, 
  SlidingWindowTagger]

for model in models:
  event.listen(model, "before_update", BaseModel._update_timestamp)

def init_db() -> None:
  for model in models:
    model.metadata.create_all(bind=engine)

def get_session() -> Session:
  return Session(engine)
