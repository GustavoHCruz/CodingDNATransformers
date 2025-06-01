
from dotenv import load_dotenv
from routes import (dataset_route, parent_dataset_route, ping_route,
                    progress_tracker_route)

load_dotenv()

from fastapi import FastAPI
from lifespan import lifespan

app = FastAPI(lifespan=lifespan)

# Routes
app.include_router(dataset_route.router)
app.include_router(parent_dataset_route.router)
app.include_router(ping_route.router)
app.include_router(progress_tracker_route.router)