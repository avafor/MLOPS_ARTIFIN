import mlflow
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np  
from typing import Optional
import sys




mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
scaler = joblib.load('scaler.joblib') # while running the Dockerfile, we copy the scaler.joblib file to the same directory as the predict.py script, so we can load it directly from there.
#scaler = joblib.load('/Users/iashahba/Documents/ARTIFIN/FS26/Data and Code/01_model_tracking/scaler.joblib')


#loaded_model = mlflow.pyfunc.load_model(MODEL_DIR.as_posix())
loaded_model = mlflow.pyfunc.load_model(
    "models:/iris_baseline/Staging"
)
def predict(input_data):
    # Always convert first
    input_data = np.array(input_data)
    # If user passes a single sample like [5.1, 3.5, 1.4, 0.2]
    # turn it into shape (1, 4)
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)

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


# To run the FastAPI run "uvicorn predict:app --reload" in your terminal 

# we can add a main block to allow for command-line predictions as well, which can be useful for testing without starting the server.
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict.py f1 f2 f3 f4")
        sys.exit(1)

    sample = [float(x) for x in sys.argv[1:]]
    input_array = np.array(sample).reshape(1, -1)

    prediction = predict(input_array)
    print("Prediction:", prediction)