import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler

# https://stackoverflow.com/questions/66056695/what-does-labelencoder-fit-do
# https://towardsdatascience.com/how-and-why-to-standardize-your-data-996926c2c832

DATASET_PATH = "Train_Test_IoT_dataset/"

def preprocess_data(file, features, labels):
  data = pd.read_csv(DATASET_PATH + file)

  data = data[features + labels]

  for column in data.select_dtypes(include=["object"]).columns:
    data[column] = LabelEncoder().fit_transform(data[column])

  scaler = StandardScaler()
  data[features] = scaler.fit_transform(data[features])

  return data[features], data[labels]
