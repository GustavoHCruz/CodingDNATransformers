from typing import Any, Optional

from pydantic import BaseModel


class BaseResponse(BaseModel):
  status: str
  message: Optional[str] = None
  data: Optional[Any] = None