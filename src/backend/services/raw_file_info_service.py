from database.db import get_session
from models.base_model import ApproachEnum
from models.raw_file_info_model import RawFileInfo
from sqlmodel import select


def get_by_file_name_and_approach(file_name: str, approach: ApproachEnum) -> RawFileInfo | None:
  with get_session() as session:
    statement = select(RawFileInfo).where(RawFileInfo.file_name == file_name).where(RawFileInfo.approach == approach)
    result = session.exec(statement).first()
    return result

def save_file(file_name: str, approach: ApproachEnum, records: int) -> RawFileInfo:
  with get_session() as session:
    new_instance = RawFileInfo(file_name=file_name, approach=approach, total_records=records)
    session.add(new_instance)
    session.commit()
    session.refresh(new_instance)
    return new_instance