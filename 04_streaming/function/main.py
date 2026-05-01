import base64
import json
import os
import tempfile

import joblib
import mlflow.pyfunc
from google.cloud import pubsub_v1
from google.cloud import storage

PROJECT_ID = os.environ.get("GCP_PROJECT")
OUTPUT_TOPIC = os.environ.get("PREDICTIONS_TOPIC", "iris-predictions")
MODEL_URI = os.environ.get("MODEL_URI")
SCALER_URI = os.environ.get("SCALER_URI")

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, OUTPUT_TOPIC)

model = None
scaler = None

CLASS_NAMES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


def download_blob_from_gcs(gcs_uri: str, local_path: str):
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    uri_without_prefix = gcs_uri[len("gs://"):]
    bucket_name, blob_name = uri_without_prefix.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)


def get_model():
    global model
    if model is None:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    return model


def get_scaler():
    global scaler
    if scaler is None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            scaler_local_path = tmp_file.name

        download_blob_from_gcs(SCALER_URI, scaler_local_path)
        scaler = joblib.load(scaler_local_path)

    return scaler


def prepare_features(record: dict):
    return [[
        record["sepal_length"],
        record["sepal_width"],
        record["petal_length"],
        record["petal_width"],
    ]]





def classify_iris(cloud_event):
    message_data = cloud_event.data["message"]["data"]
    decoded = base64.b64decode(message_data).decode("utf-8")
    event = json.loads(decoded)

    print("Function triggered successfully")
    print(event)    