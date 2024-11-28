from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time
import joblib
from preprocessing_data import preprocess_data

def train_models(devices):
  for device, item in devices.items():
    features, labels = preprocess_data(item["train_test_file"], item["features"], item["labels"])

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Random Forest Classifier
    start = time.time()
    model = RandomForestClassifier(class_weight="balanced")
    model.fit(X_train, y_train)
    end = time.time()

    joblib.dump(model, f"models/{device}_random_forest_classifier.pkl")
    print(f"Random Forest Classifier for {device} trained in {end - start:.2f} seconds")

    # Logistic Regression
    start = time.time()
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)
    end = time.time()

    joblib.dump(model, f"models/{device}_logistic_regression.pkl")
    print(f"Logistic Regression for {device} trained in {end - start:.2f} seconds")

    # SVM
    start = time.time()
    model = SVC(probability=True, class_weight="balanced")
    model.fit(X_train, y_train)
    end = time.time()

    joblib.dump(model, f"models/{device}_svm.pkl")
    print(f"SVM for {device} trained in {end - start:.2f} seconds")

    joblib.dump((X_test, y_test), f"models/{device}_test_data.pkl")