from pathlib import Path
import os
import tempfile

import joblib
import mlflow
import mlflow.artifacts
import mlflow.sklearn
import numpy as np
import pandas as pd

from prefect import flow, task, get_run_logger

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "Iris Baseline Experiment"


@task
def load_data() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.rename(columns={"target": "target"}, inplace=True)
    return df


@task
def check_data_quality(df: pd.DataFrame) -> pd.Series:
    logger = get_run_logger()
    missing = df.isnull().sum()
    logger.info(f"Missing values:\n{missing}")
    return missing


@task
def split_data(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


@task
def train_and_log_model(X_train, X_val, X_test, y_train, y_val, y_test) -> str:
    logger = get_run_logger()

    os.makedirs("./mlruns", exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    solver = "saga"

    with mlflow.start_run() as run:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            random_state=42,
            max_iter=200,
            solver=solver,
        )
        model.fit(X_train_scaled, y_train)

        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)

        report = classification_report(y_test, y_test_pred, output_dict=True)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("solver", solver)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", 200)

        mlflow.log_metric("val_accuracy", accuracy_score(y_val, y_val_pred))
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("test_precision_macro", report["macro avg"]["precision"])
        mlflow.log_metric("test_recall_macro", report["macro avg"]["recall"])
        mlflow.log_metric("test_f1_macro", report["macro avg"]["f1-score"])

        mlflow.sklearn.log_model(model, artifact_path="model")

        # save scaler and log it as artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = Path(tmpdir) / "scaler.joblib"
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")

        run_id = run.info.run_id
        logger.info(f"Finished training. run_id={run_id}")

    return run_id


@task
def generate_batch(df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
    features = df.drop(columns=["target"])
    batch = features.sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)
    return batch


@task
def batch_predict(run_id: str, batch_df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

    # download scaler artifact from the same MLflow run
    scaler_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/preprocessing"
    )
    scaler = joblib.load(Path(scaler_dir) / "scaler.joblib")

    X_scaled = scaler.transform(batch_df.values)
    preds = model.predict(X_scaled)

    result = batch_df.copy()
    result["prediction"] = preds

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"batch_predictions_{run_id}.csv"
    result.to_csv(out_file, index=False)

    logger.info(f"Saved batch predictions to {out_file}")
    return result


@flow(name="iris_monthly_train_and_batch_predict")
def iris_monthly_train_and_batch_predict(n_samples: int = 100) -> str:
    df = load_data()
    check_data_quality(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    run_id = train_and_log_model(X_train, X_val, X_test, y_train, y_val, y_test)
    batch = generate_batch(df, n_samples=n_samples)
    batch_predict(run_id, batch)
    return run_id


if __name__ == "__main__":
    iris_monthly_train_and_batch_predict()