from fastapi import APIRouter, Request

from helpers.config import get_settings
from .shemes import SensorsData
from models import SensorsDataModel
from models.enums.ResponseMassages import ResponseMassages as rs
from controllers import MachineFailureController
from stores.llms.providers import CohereProvider
from models import NLPDataModel
from .shemes.givenData import ChatMassage

# for testing purposes
settings = get_settings()



nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1"]
)

@nlp_router.get("/predict") # this api to test the integration of the classic ML models
async def push_sensors_data(request: Request):
    
    # step 1: get data from Sensors
    
    # Get sensor model
    sensor_data_model = await SensorsDataModel.create_instance(
        db_client=request.app.db_client
    )

    last_updates = await sensor_data_model.get_sensors_data()

    if not last_updates:
        return {
            "message": rs.NO_FEATCHED_DATA.value
        }
    
    # step 2: preprocess data
    
    # TODO: preprocess the data as per the NLP model's requirements
    
    # step 3: predict using classic ML
    
    prediction_controller = MachineFailureController()
    
    prediction = prediction_controller.predict(last_updates)
    
    # step 4: return prediction
    
    return {
         "message": rs.PREDICTION_classic_SUCCESS.value,
         "prediction": prediction
    }


    

@nlp_router.post("/chat/{user_id}", response_model = ChatMassage) # this api to test the integration of the classic ML models
async def chat(request: Request, user_id: str, chat_massage: ChatMassage):
    # all in the app
    
    # start with cohere Now -> we will work with any coming provider but now only for testing
    
    if not hasattr(request.app, 'provider') or request.app.provider is None:
        request.app.provider = CohereProvider(
            api_key= settings.COHERE_API_KEY,
            temperature = settings.TEMPERATURE
        )
        request.app.provider.set_model(model_name = settings.MODEL_ID)
    
    if not hasattr(request.app, 'summary') or request.app.summary is None:
        nlp_data_model = await NLPDataModel.create_instance(
            db_client=request.app.db_client
        )
        
    if not hasattr(request.app,'user_id') or request.app.user_id is None:
        request.app.user_id = user_id
        
        request.app.summary_scheme = await nlp_data_model.get_chatSummary(user_id=user_id)
        
        request.app.nlp_data_model = nlp_data_model # save instead of creating another instance in the main
    
    
    response =  await request.app.provider.chat( 
                              query = chat_massage.message, 
                              summary = request.app.summary_scheme.summary
                              )
    
    
    
    return ChatMassage(
        message= response
    )


