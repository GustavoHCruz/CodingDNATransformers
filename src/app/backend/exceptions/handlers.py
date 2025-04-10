from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.backend.models.response import BaseResponse


def http_exception_handler(_: Request, exc: HTTPException):
  return JSONResponse(
    status_code=exc.status_code,
    content=BaseResponse(
      status="error",
      message=exc.detail,
      data=None
    ).model_dump()
  )

def generic_exception_handler(_: Request, exc: Exception):
  return JSONResponse(
    status_code=500,
    content=BaseResponse(
      status="error",
      message="Internal error",
      data={"exception": str(exc)}
    ).model_dump()
  )