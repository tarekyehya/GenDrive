# src/controllers/prediction_controller.py

from models import SensorsDataModel
from models.random_forest_model import RandomForestModel
import pandas as pd

class PredictionController:
    def __init__(self, db_client):
        self.db_client = db_client
        self.sensor_data_model = SensorsDataModel(db_client)
        self.model = RandomForestModel()
        # Load the pre-trained model
        self.model.load_model('src/models/random_forest_model.pkl')

    async def predict(self):
        # Fetch the latest sensor data
        latest_data = await self.sensor_data_model.get_sensors_data()
        if not latest_data:
            raise ValueError("No sensor data available for prediction.")

        # Convert the sensor data into a DataFrame
        data_df = pd.DataFrame([{
            "Air temperature [K]": latest_data.air_temperature,
            "Process temperature [K]": latest_data.process_temperature,
            "Rotational speed [rpm]": latest_data.rotational_speed,
            "Torque [Nm]": latest_data.torque,
            "Tool wear [min]": latest_data.tool_wear
        }])

        # Make predictions using the loaded model
        predictions = self.model.predict(data_df)
        return predictions[0]