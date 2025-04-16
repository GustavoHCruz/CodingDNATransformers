from typing import List, Optional

from pydantic import BaseModel


class CreationSettings(BaseModel):
	genbank: Optional[bool] = None
	gencode: Optional[bool] = None
	approachs: Optional[List[bool]] = None