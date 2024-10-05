from fastapi import FastAPI, APIRouter, Request
from helpers.config import get_settings, Settings
from .shemes import SensorsData
from models import SensorsDataModel
from models.enums.ResponseMassages import ResponseMassages as rs


sensor_router = APIRouter(
    prefix="/api/v1/sensors",
    tags=["api_v1"]

)

@sensor_router.post("/addData")
async def PushSensorsData(request : Request,data : SensorsData):
    
    # Get sensor model
    sensor_data_model = await SensorsDataModel.create_instance(
        db_client=request.app.db_client
    )
    
    # Add data to database
    is_inserted = await sensor_data_model.insert_sensors_data(data=data)
    
    if  not is_inserted:
        return {
            "message": rs.DB_SENSORS_DATA_FAILED_UPLODED.value
            }
    
    # test get data from database
    #result = await sensor_data_model.get_sensors_data()
    #return result
    
    # Return success response
    return {
        "message": rs.DB_SENSORS_DATA_SUCCESS_UPLODED.value
        }
    
    
    
    


    
