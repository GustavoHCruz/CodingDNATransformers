from typing import Optional

from models.base_model import ApproachEnum
from models.raw_file_info_model import RawFileInfo
from services.decorators import with_session
from sqlmodel import Session, select


@with_session
def get_by_file_name_and_approach(file_name: str, approach: ApproachEnum, session: Optional[Session] = None) -> RawFileInfo | None:
  stmt = select(RawFileInfo).where(RawFileInfo.file_name == file_name).where(RawFileInfo.approach == approach)
  result = session.exec(stmt).first()
  return result

@with_session
def save_file(file_name: str, approach: ApproachEnum, records: int, session: Optional[Session] = None) -> RawFileInfo:
  new_instance = RawFileInfo(file_name=file_name, approach=approach, total_records=records)
  session.add(new_instance)
  session.commit()
  session.refresh(new_instance)
  return new_instance