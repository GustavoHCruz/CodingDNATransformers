from typing import Generator, List

from models.base_model import ApproachEnum
from models.parent_record_model import ParentRecord

seen_global = set()

def remove_duplicated(instances: List[ParentRecord]) -> List[ParentRecord]:
  global seen_global
  unique = []

  key_fields = ["sequence", "target", "flank_before", "flank_after", "organism", "gene"]

  for instance in instances:
    key = tuple(instance.get(field, "") for field in key_fields)

    if key not in seen_global:
      seen_global.add(key)
      unique.append(instance)
  
  return unique