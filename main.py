from train_model import train_models
from test_model import test_models

def main():
  devices = {
    "fridge": {
      "train_file": "Train_Test_IoT_Fridge.csv",
      "test_file": "",
      "features": ["fridge_temperature", "temp_condition"],
      "labels": ["label"]
    },
    "garage_door": {
      "train_file": "Train_Test_IoT_Garage_Door.csv",
      "test_file": "",
      "features": ["door_state", "sphone_signal"],
      "labels": ["label"]
    },
    "gps_tracker": {
      "train_file": "Train_Test_IoT_GPS_Tracker.csv",
      "test_file": "",
      "features": ["latitude", "longitude"],
      "labels": ["label"]
    },
    "modbus": {
      "train_file": "Train_Test_IoT_Modbus.csv",
      "test_file": "",
      "features": ["FC1_Read_Input_Register", "FC2_Read_Discrete_Value", "FC3_Read_Holding_Register", "FC4_Read_Coil"],
      "labels": ["label"]
    },
    "motion_light": {
      "train_file": "Train_Test_IoT_Motion_Light.csv",
      "test_file": "",
      "features": ["motion_status", "light_status"],
      "labels": ["label"]
    },
    "thermostat": {
      "train_file": "Train_Test_IoT_Thermostat.csv",
      "test_file": "",
      "features": ["current_temperature", "thermostat_status"],
      "labels": ["label"]
    },
    "weather": {
      "train_file": "Train_Test_IoT_Weather.csv",
      "test_file": "",
      "features": ["temperature", "pressure", "humidity"],
      "labels": ["label"]
    }
  }

  train_models(devices)
  test_models(devices)
