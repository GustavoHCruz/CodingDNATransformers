
from dotenv import load_dotenv
from routes import datasets_route, ping_route, progress_tracker_route

load_dotenv()

from fastapi import FastAPI, HTTPException
from handlers.exception_handlers import (generic_exception_handler,
                                         http_exception_handler)
from lifespan import lifespan

app = FastAPI(lifespan=lifespan)

# Global Handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Routes
app.include_router(ping_route.router)
app.include_router(datasets_route.router)
app.include_router(progress_tracker_route.router)