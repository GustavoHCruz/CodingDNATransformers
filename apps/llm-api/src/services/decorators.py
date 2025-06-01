from functools import wraps
from typing import Callable, Optional, ParamSpec, TypeVar, cast

from database.db import get_session
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlmodel import Session
from starlette.status import (HTTP_200_OK, HTTP_400_BAD_REQUEST,
                              HTTP_500_INTERNAL_SERVER_ERROR)

P = ParamSpec("P")
R = TypeVar("R")

def standard_response(
	default_status_code: int = HTTP_200_OK,
	default_message: str = "OK"
) -> Callable[[Callable[P, object]], Callable[P, JSONResponse]]:
	def decorator(func: Callable[P, object]) -> Callable[P, JSONResponse]:
		@wraps(func)
		def wrapper(*args: P.args, **kwargs: P.kwargs) -> JSONResponse:
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

def with_session(func: Callable[P, R]) -> Callable[P, R]:
	@wraps(func)
	def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
		session = cast(Optional[Session], kwargs.get("session"))
		if session is not None:
			return func(*args, **kwargs)
		
		with get_session() as session:
			kwargs["session"] = session
			return func(*args, **kwargs)

	return wrapper