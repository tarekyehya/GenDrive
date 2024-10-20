from fastapi import FastAPI
from routes import base, sensors, nlp
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings

from controllers.NLPController import NLPController

@asynccontextmanager
async def lifespan(app: FastAPI): # for comming actions
    # code here will run before starting
    # add to app what you need to access in the full project
    # Making the connection
    settings = get_settings()
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]
    
    
    yield
    # code here will run after shutdown
    nlp_controller = NLPController(
        DataModelObject= app.nlp_data_model,
        providerObject= app.provider,
        user_id = app.user_id, 
        )
    
    await nlp_controller.manage_history()
    
    # manage chat history
    
    
    app.mongo_conn.close()


app = FastAPI(lifespan = lifespan)

app.include_router(base.base_router)
app.include_router(sensors.sensor_router)
app.include_router(nlp.nlp_router)
