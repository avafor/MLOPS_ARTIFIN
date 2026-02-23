import mlflow
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np  
from typing import Optional
from pathlib import Path


mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

scaler = joblib.load('/Users/iashahba/Documents/ARTIFIN/FS26/Data and Code/01_model_tracking/scaler.joblib')

MODEL_DIR = Path(
    "/Users/iashahba/Documents/ARTIFIN/FS26/Data and Code/01_model_tracking/mlruns/"
    "316585724729739867/models/m-a44d65f42a0f4e54a4d05dc9c30409cd/artifacts"
)
loaded_model = mlflow.pyfunc.load_model(MODEL_DIR.as_posix())
def predict(input_data):

    # apply the scaler to the input data
    input_data = scaler.transform(input_data)
    return loaded_model.predict(input_data)


app = FastAPI()

class PredictRequest(BaseModel):
    run_id: Optional[str] = None
    input_data: list[float]

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    input_array = np.array(req.input_data).reshape(1, -1)
    prediction = predict(input_array)
    return {"prediction": prediction.tolist()}

