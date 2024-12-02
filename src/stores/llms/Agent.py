
from langchain_core.tools import Tool

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import Runnable

from .providers.llm_shemes.MassageScheme import State
from controllers.MachineFailureController import MachineFailureController
from models import SensorsDataModel
from models.enums.ResponseMassages import ResponseMassages as rs



class Agent:
    def __init__(self,db_client = None):
        self.db_client = db_client
    
    def get_tools(self):
        return [
            Tool(
                func=self.components_health,
                 name="components_health",
                 description="Tool to check the car's components healthy."
                 ),
        ]
            
    # classic tool 
    async def _classic_engine_health(self)-> int:
        
        # step 1: get data from Sensors
    
        # Get sensor model
        sensor_data_model = await SensorsDataModel.create_instance(
            db_client= self.db_client
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
        
        return prediction
    
    
    async def components_health(self):
        
        """
            Tool to check the car's components healthy.
            

            Returns:
                string: the statues of the car.
        """
        
        # TODO: implement the logic to check components health
        
        
        # classic for only the engine
        
        prediction_engine = await self._classic_engine_health()
        
        if prediction_engine == 1:
            prediction_engine_massage = rs.PREDICTION_ENGINE_HEALTH_BAD.value
        else:
            prediction_engine_massage = rs.PREDICTION_ENGINE_HEALTH_GOOD.value
        
        print(prediction_engine_massage)
        
        return prediction_engine_massage # for now
    
  #  @tool
    async def maintenance_schedule(self):
        
        # TODO: implement the logic to generate maintenance schedule
        
        # TODO: generate a maintenance schedule based on the components health prediction
        pass
        
#    @tool
    async def repair_instructions(self):
        
        # TODO: implement the logic to generate repair instructions
        
        # TODO: generate repair instructions based on the components health prediction
        
        pass
    
    
        
