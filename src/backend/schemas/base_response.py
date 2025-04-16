from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")

class BaseResponse(BaseModel, Generic[T]):
  status: str
  message: Optional[str] = None
  data: Optional[T] = None