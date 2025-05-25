from functools import wraps
from typing import Callable, Optional

from database.db import get_session
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlmodel import Session
from starlette.status import (HTTP_200_OK, HTTP_400_BAD_REQUEST,
                              HTTP_500_INTERNAL_SERVER_ERROR)


def standard_response(default_status_code: int = HTTP_200_OK, default_message: str = "OK"):
	def decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			try:
				result = func(*args, **kwargs)
				return JSONResponse(
					status_code=default_status_code,
					content={
						"status": "success",
						"message": default_message,
						"data": jsonable_encoder(result)
					}
				)
			except HTTPException as e:
				return JSONResponse(
					status_code=e.status_code,
					content={
						"status": "error",
						"message": e.detail
					}
				)
			except ValueError as e:
				return JSONResponse(
					status_code=HTTP_400_BAD_REQUEST,
					content={
						"status": "error",
						"message": str(e)
					}
				)
			except Exception as e:
				return JSONResponse(
					status_code=HTTP_500_INTERNAL_SERVER_ERROR,
					content={
						"status": "error",
						"message": str(e)
					}
				)
		return wrapper
	return decorator

def with_session(func: Callable):
	@wraps(func)
	def wrapper(*args, session: Optional[Session] = None, **kwargs):
		if session is not None:
			return func(*args, session=session, **kwargs)
		
		with get_session() as session:
			return func(*args, session=session, **kwargs)

	return wrapper