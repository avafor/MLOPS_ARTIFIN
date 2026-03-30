import os
import uuid
import joblib
import mlflow
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

RAW_FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

CLEAN_FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def main():
    os.makedirs("data/current_batches", exist_ok=True)

    # load scaler and model
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    scaler = joblib.load("scaler.joblib")
    model = mlflow.pyfunc.load_model("models:/iris_baseline/Staging")

    # load full Iris data
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()

    # create one batch by sampling rows
    batch = X.copy()
    batch["target"] = y
    batch = batch.sample(n=30, replace=True).reset_index(drop=True)

    # optional drift for demonstration
    batch["petal length (cm)"] = batch["petal length (cm)"] + 1.0

    # predict on scaled features
    X_batch = batch[RAW_FEATURES]
    X_scaled = scaler.transform(X_batch)
    X_scaled = pd.DataFrame(X_scaled, columns=RAW_FEATURES)

    predictions = model.predict(X_scaled)

    # save batch with clean column names
    batch.columns = CLEAN_FEATURES + ["target"]
    batch["prediction"] = predictions

    batch_id = str(uuid.uuid4())[:8]
    batch.to_csv(f"data/current_batches/{batch_id}.csv", index=False)

    print(f"batch saved: {batch_id}.csv")


if __name__ == "__main__":
    main()