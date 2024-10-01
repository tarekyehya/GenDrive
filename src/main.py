from fastapi import FastAPI
from routes import base
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI): # for comming actions
    # code here will run before starting
    # add to app what you need to access in the full project
    yield
    # code here will run after shutdown


app = FastAPI(lifespan = lifespan)

app.include_router(base.base_router)
