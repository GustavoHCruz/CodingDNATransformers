from functools import wraps
from typing import Callable, Optional

from database.db import get_session
from fastapi import HTTPException
from sqlmodel import Session


def handle_exceptions(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		try:
			return func(*args, **kwargs)
		except HTTPException:
			raise
		except Exception as e:
			raise HTTPException(status_code=500, detail=str(e))
	return wrapper

def with_session(func: Callable):
	@wraps(func)
	def wrapper(*args, session: Optional[Session] = None, **kwargs):
		own_session = False
		if session is None:
			session = get_session()
			own_session = True
		
		try:
			return func(*args, session=session, **kwargs)
		finally:
			if own_session:
				session.close()
		
	return wrapper