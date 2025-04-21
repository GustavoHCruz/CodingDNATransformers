from database.db import get_session
from models.raw_file_info_model import RawFileInfo
from sqlmodel import select


def get_by_file_name(file_name: str) -> RawFileInfo | None:
  with get_session() as session:
    statement = select(RawFileInfo).where(RawFileInfo.file_name == file_name)
    result = session.exec(statement).first()
    return result