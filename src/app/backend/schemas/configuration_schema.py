from typing import Optional

from pydantic import BaseModel


class ConfigPatch(BaseModel):
  datasets_raw_dir: Optional[str] = None
  datasets_dir: Optional[str] = None
  cache_dir: Optional[str] = None
  cache_file: Optional[str] = None
  genbank_file: Optional[str] = None
  genbank_file: Optional[str] = None
  gencode_file_annotations: Optional[str] = None