"""Columns Understanding 
Type: consisting of a letter L, M, or H for low, medium and high as product quality variants.

air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K.

process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.

rotational speed [rpm]: calculated from a power of 2860 W, overlaid with a normally distributed noise.

torque [Nm]: torque values are normally distributed around 40 Nm with a Ïƒ = 10 Nm and no negative values.

tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.

machine failure: whether the machine has failed in this particular datapoint for any of the following failure modes are true.

The machine failure consists of five independent failure modes

tool wear failure (TWF): the tool will be replaced of fail at a randomly selected tool wear time between 200 ~ 240 mins.

heat dissipation failure (HDF): heat dissipation causes a process failure, if the difference between air and process temperature is below 8.6 K and the rotational speed is below 1380 rpm

power failure (PWF): the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails.

overstrain failure (OSF): if the product of tool wear and torque exceeds 11,000 minNm for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain.

random failures (RNF): each process has a chance of 0,1 % to fail regardless of its process parameters."""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sklearn.metrics as sk_metrics
from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib


'''MachineFailureData: Handles data loading, preprocessing, and splitting.'''
class MachineFailureData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def preprocess_data(self):
        non_numeric_columns = self.data.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        for column in non_numeric_columns:
            self.data[column] = label_encoder.fit_transform(self.data[column])
        self.data=self.data.drop(['TWF','HDF','PWF','OSF','RNF','id'],axis=1)


    def balance_data(self):
        train_data_failed = self.data[self.data['Machine failure'] == 1]
        num_samples = train_data_failed['Machine failure'].count()
        train_data_notfailed = self.data[self.data['Machine failure'] == 0]
        train_data_notfailed_balanced = train_data_notfailed.sample(n=num_samples, random_state=42)
        balanced_train_data = pd.concat([train_data_notfailed_balanced, train_data_failed])
        return balanced_train_data.sample(frac=1, random_state=42).reset_index(drop=True)

    def split_data(self, balanced_data):
        y = balanced_data['Machine failure']
        x = balanced_data.drop(['Machine failure', 'Product ID', 'Type'], axis=1)
        return train_test_split(x, y, random_state=1, test_size=0.3)

'''RandomForestModel: Manages the Random Forest model, including training and evaluation.'''

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

    def load_model(self, file_path):
        self.model = joblib.load(file_path)
 


'''MachineFailureController: Coordinates the data processing and model operations, including visualization.'''

class MachineFailureController:
    def __init__(self, file_path, model_path=None):
        self.data_model = MachineFailureData(file_path)
        self.rf_model = RandomForestModel()
        self.model_path = model_path

    def run(self, train_model=True):
        self.data_model.load_data()
        self.data_model.preprocess_data()

        if train_model:
            balanced_data = self.data_model.balance_data()
            x_train, x_test, y_train, y_test = self.data_model.split_data(balanced_data)

            self.rf_model.train(x_train, y_train)

            # Save the model using the specified path or default path
            save_path = self.model_path if self.model_path else "default_model.pkl"
            self.rf_model.save_model(save_path)

            y_pred = self.rf_model.predict(x_test)
            accuracy, conf_matrix = self.rf_model.evaluate(y_test, y_pred)
            print("Accuracy:", accuracy)
            print("Confusion Matrix:")
            print(conf_matrix)
            self.visualize_confusion_matrix(conf_matrix)

            # Validation
            validation_data = self.data_model.data.sample(n=10000, random_state=42)
            y_validation = validation_data['Machine failure']
            x_validation = validation_data.drop(['Machine failure', 'Product ID', 'Type'], axis=1)
            y_pred_val = self.rf_model.predict(x_validation)
            accuracy_val, conf_matrix_val = self.rf_model.evaluate(y_validation, y_pred_val)
            print("Validation Accuracy:", accuracy_val)
            print("Validation Confusion Matrix:")
            print(conf_matrix_val)
            self.visualize_confusion_matrix(conf_matrix_val)
        else:
            # Load the model only if a model_path is provided
            if self.model_path:
                self.rf_model.load_model(self.model_path)
            else:
                raise ValueError("Model path must be provided to load a pre-trained model.")

    def predict_new_data(self, new_data):
        return self.rf_model.predict(new_data)

    def visualize_confusion_matrix(self, conf_matrix):
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=self.rf_model.model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()


def preprocess_data_for_predictions(csv_file, columns=["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Check if all columns exist in the DataFrame
        if not all(col in df.columns for col in columns):
            raise ValueError("One or more columns do not exist in the CSV file.")

        # Select the specified columns
        selected_df = df[columns]

        return selected_df

    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return None

    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file}' is empty.")
        return None

    except pd.errors.ParserError:
        print(f"Error: An error occurred while parsing the file '{csv_file}'.")
        return None




'''
if __name__ == "__main__":
    controller = MachineFailureController(r"src\datasets\machine_failure\train.csv", r"src\models\random_forest_model.pkl")
    controller.run()


'''
if __name__ == "__main__":
    controller = MachineFailureController(r"src\datasets\machine_failure\train.csv", r"src\models\random_forest_model.pkl")
    controller.run(train_model=False)

    # Example of predicting new data

    # Define the column names used during training
    columns = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

    # Create a DataFrame for a single input
    new_data_single_input = pd.DataFrame({
        "Air temperature [K]": [302.4],
        "Process temperature [K]": [311.8],
        "Rotational speed [rpm]": [1509],
        "Torque [Nm]": [45.7],
        "Tool wear [min]": [108]
    })

    predictions = controller.predict_new_data(new_data_single_input)
    print("Predicted class for the single input:", predictions)
    #new_data=preprocess_data_for_predictions(r"D:\Car_Assistant\GenDrive\src\datasets\machine_failure\test.csv")
   # predictions = controller.predict_new_data(new_data)
   # print(predictions)





 
      