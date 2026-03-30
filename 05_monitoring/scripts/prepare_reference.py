import os
import joblib
import mlflow
import pandas as pd
from sklearn.datasets import load_iris

# These are the original column names used when the scaler was trained
RAW_FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# These are cleaner names for monitoring and dashboards
CLEAN_FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def main():
    # create folder if it does not exist
    os.makedirs("data", exist_ok=True)

    # connect to MLflow and load artifacts
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    scaler = joblib.load("scaler.joblib")
    model = mlflow.pyfunc.load_model("models:/iris_baseline/Staging")

    # load Iris data
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()

    # scale features before prediction
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=RAW_FEATURES)

    # generate predictions
    predictions = model.predict(X_scaled)

    # save raw data + target + prediction as reference dataset
    reference = X.copy()
    reference.columns = CLEAN_FEATURES
    reference["target"] = y
    reference["prediction"] = predictions

    reference.to_csv("data/reference.csv", index=False)
    print("reference.csv created")


if __name__ == "__main__":
    main()