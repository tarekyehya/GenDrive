from pydantic import BaseModel
from typing import Optional

class SensorsData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    oil: float
    gas: float
    
    # can add more properties in the future

class SummaryChat(BaseModel):
    user_id : str = 0 # we may need to make the scheme more tolerant, and we can update the user_id
    summary : str = ""
    

class ChatMassage(BaseModel):
    message : str
    
