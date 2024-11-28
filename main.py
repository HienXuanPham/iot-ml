from train_model import train_models
from test_model import test_models

def main():
  devices = {
    "fridge": {
      "train_test_file": "Train_Test_IoT_Fridge.csv",
      "features": ["fridge_temperature", "temp_condition"],
      "labels": ["label"]
    },
    "garage_door": {
      "train_test_file": "Train_Test_IoT_Garage_Door.csv",
      "features": ["door_state", "sphone_signal"],
      "labels": ["label"]
    },
    "gps_tracker": {
      "train_test_file": "Train_Test_IoT_GPS_Tracker.csv",
      "features": ["latitude", "longitude"],
      "labels": ["label"]
    },
    "modbus": {
      "train_test_file": "Train_Test_IoT_Modbus.csv",
      "features": ["FC1_Read_Input_Register", "FC2_Read_Discrete_Value", "FC3_Read_Holding_Register", "FC4_Read_Coil"],
      "labels": ["label"]
    },
    "motion_light": {
      "train_test_file": "Train_Test_IoT_Motion_Light.csv",
      "features": ["motion_status", "light_status"],
      "labels": ["label"]
    },
    "thermostat": {
      "train_test_file": "Train_Test_IoT_Thermostat.csv",
      "features": ["current_temperature", "thermostat_status"],
      "labels": ["label"]
    },
    "weather": {
      "train_test_file": "Train_Test_IoT_Weather.csv",
      "features": ["temperature", "pressure", "humidity"],
      "labels": ["label"]
    }
  }

  X_test, y_test = train_models(devices)
  test_models(X_test, y_test)

if __name__ == "__main__":
  main()