from database.db import get_session
from models.exin_classifier_model import ExInClassifier
from services.decorators import handle_exceptions


def remove_duplicated_hashs(instances: list[ExInClassifier]) -> list[ExInClassifier]:
  seen = set()
  unique = []

  for inst in instances:
    if inst.hash_id not in seen:
      seen.add(inst.hash_id)
      unique.append(inst)
  
  return unique

@handle_exceptions
def bulk_create(instances: list[ExInClassifier]) -> bool:
  with get_session() as session:
    if not instances:
      return False
    
    filtered_data = (remove_duplicated_hashs(instances))

    session.add_all(filtered_data)
    session.commit()

    return True