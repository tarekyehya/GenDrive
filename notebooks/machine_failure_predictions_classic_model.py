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
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sklearn.metrics as sk_metrics
from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import random
# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback

current_directory=os.path.dirname(os.path.abspath("machine_failure_predictions_classic_model.ipynb"))
if os.name=="nt":
    current_directory=current_directory[0].upper()+current_directory[1:]
  
current_directory = current_directory.split(os.sep)

# Remove the unwanted part (e.g., 'Projects')
current_directory = [part for part in current_directory if part != 'notebooks']

# Reconstruct the path
current_directory = os.sep.join(current_directory)

train_path=os.path.join(current_directory,r"src\datasets\machine_failure\train.csv")
test_path=os.path.join(current_directory,r"src\datasets\machine_failure\test.csv")
sample_path=os.path.join(current_directory,r"src\datasets\machine_failure\sample_submission.csv")


train_path=r"{}".format(train_path)
test_path=r"{}".format(test_path)
sample_path=r"{}".format(sample_path)

def replace_slashes(path, to_forward=True):
    if to_forward:
        return path.replace("\\", "/")
    else:
        return path.replace("/", "\\")
if os.name=="nt":
    train_path = replace_slashes(train_path, to_forward=True)
    test_path = replace_slashes(test_path, to_forward=True)
    sample_path = replace_slashes(sample_path, to_forward=True)


train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
sample_data=pd.read_csv(sample_path)

# convert the numerical columns to float for consistency
int_cols = ['Rotational speed [rpm]', 'Tool wear [min]']
train_data[int_cols] = train_data[int_cols].astype(np.float64)
test_data[int_cols] = test_data[int_cols].astype(np.float64)

# make a copy of training data for modelling
training = train_data.copy()
training.drop(['id','Product ID'], axis=1, inplace=True)

# make a copy of test data for predictions
testing = test_data.copy()
testing.drop(['id','Product ID'], axis=1, inplace=True)

# encoding the only categorical column
le = LabelEncoder()
training['Type'] = le.fit_transform(training['Type'])
testing['Type'] = le.fit_transform(testing['Type'])

training_float = training.select_dtypes(include=[float])
training_float['Machine failure'] = training['Machine failure']


# normalise the data
norm = Normalizer()
feature_float_cols = training_float.columns[0:5]

#for c in float_value_cols:
#    temp_array = norm.fit_transform([training[c]])
#    temp_list = temp_array.flatten().tolist()
#    training[c] = temp_list

# categorical columns for one-hot-encoding
feature_cat_cols = ['Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']


# create a preprocessing pipeline
transform_float = make_pipeline(Normalizer())
transform_cat = make_pipeline(OneHotEncoder(handle_unknown='ignore'))

preprocessor = make_column_transformer((transform_float, feature_float_cols),
                                       (transform_cat, feature_cat_cols),   
                                      )
# feature engineering
x, y = training.drop('Machine failure',axis=1), training[['Machine failure']]


# train - validation split 
random_state = random.randint(1,100)
test_size = 0.25 
shuffle = True 
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=y)

# normalize & one-hot encode the split data 
x_train = preprocessor.fit_transform(x_train)
x_val = preprocessor.transform(x_val)

input_shape = x_train.shape[1]
print(f"training data size: {x_train.shape[0]}\nvalidation data size: {x_val.shape[0]}")



# define model
binary_clf_model = keras.Sequential([layers.Dense(256, input_shape=(input_shape,), activation='relu'),
                                     layers.BatchNormalization(),
                                     layers.Dropout(0.3),
                                     layers.Dense(256, activation='relu'), 
                                     layers.BatchNormalization(),
                                     layers.Dropout(0.3),
                                     layers.Dense(1, activation='sigmoid'),])


# compile model
binary_clf_model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['binary_accuracy'],)


# model summary
binary_clf_model.summary()



size = 128
epochs = 100

early_stopping = keras.callbacks.EarlyStopping(patience=5,
                                               min_delta=0.001,
                                               restore_best_weights=True,)

tf.keras.backend.clear_session()
history = binary_clf_model.fit(x_train, y_train,
                               validation_data=(x_val, y_val),
                               batch_size=size,
                               epochs=epochs,
                               verbose=0,
                               callbacks=[early_stopping, TqdmCallback(verbose=0)],)


history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")


# preprocessing the testing data
testing_pp = preprocessor.fit_transform(testing)


predictions = binary_clf_model.predict(testing_pp)
# round predictions 
rounded = [round(x[0]) for x in predictions]




