from sklearn.metrics import classification_report, precision_recall_curve, PrecisionRecallDisplay
import time
import joblib
import matplotlib.pyplot as plt

# https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248
# https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html

MODELS = {
  "Random Forest Classifier": "random_forest_classifier",
  "Logistic Regression": "logistic_regression",
  "SVM": "svm"
}

def test_models(devices):
  figure, axs = plt.subplots(len(devices), 1, figsize=(5, 5))
  for _, (device, ax) in enumerate(zip(devices.keys(), axs)):
    X_test, y_test = joblib.load(f"models/{device}_test_data.pkl")

    print("-------------------------------------------------------")
    print(f"Testing models for {device}")

    for model_name in MODELS.values():
      model_file = f"models/{device}_{model_name}.pkl"
      model = joblib.load(model_file)

      start = time.time()
      y_predict = model.predict(X_test)
      end = time.time()

      print(f"{model} tested in {end - start:.2f} seconds")
      print(classification_report(y_test, y_predict))

      y_probability = model.predict_proba(X_test)[:, 1]
      precision, recall, _ = precision_recall_curve(y_test, y_probability)

      display = PrecisionRecallDisplay(precision=precision, recall=recall)
      display.plot(ax=ax, name=model_name)

    ax.set_title(f"Precison-Recall Curve for {device}")

  plt.show()