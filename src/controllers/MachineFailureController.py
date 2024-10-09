# src/controllers/prediction_controller.py
from .BaseController import BaseController
from routes.shemes.givenData import SensorsData

import os
import joblib
import pandas as pd


class MachineFailureController(BaseController):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(self.assets_dir,self.app_settings.MODEL_CLASSIC_PATH)
        self.load_model(self.model_path)

    def predict(self, latest_data: SensorsData): # TODO: any preprocess needed here?
        # Fetch the latest sensor data
        #latest_data = await self.sensor_data_model.get_sensors_data()
        #if not latest_data:
        #    raise ValueError("No sensor data available for prediction.")  ==> in the route, we do not have a db client connection here

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
        return int(predictions[0])
    
    def load_model(self, file_path):
        self.model = joblib.load(file_path)

"""
class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, conf_matrix

    def save_model(self, file_path="default_model.pkl"):
        joblib.dump(self.model, file_path)

    
        """