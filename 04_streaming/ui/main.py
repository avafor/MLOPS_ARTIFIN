import json
import os
import time
import uuid

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google.cloud import pubsub_v1

app = FastAPI()
templates = Jinja2Templates(directory="templates")

PROJECT_ID = os.environ["PROJECT_ID"]
INPUT_TOPIC = os.environ.get("INPUT_TOPIC", "iris-features")
PREDICTION_SUBSCRIPTION = os.environ.get(
    "PREDICTION_SUBSCRIPTION",
    "iris-ui-predictions-sub"
)

publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()

topic_path = publisher.topic_path(PROJECT_ID, INPUT_TOPIC)
subscription_path = subscriber.subscription_path(
    PROJECT_ID,
    PREDICTION_SUBSCRIPTION
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    flower_id = str(uuid.uuid4())

    message = {
        "flower_id": flower_id,
        "features": {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
    }

    publisher.publish(
        topic_path,
        json.dumps(message).encode("utf-8")
    )

    prediction = wait_for_prediction(flower_id)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction
        }
    )

def wait_for_prediction(flower_id: str, timeout: int = 10):
    deadline = time.time() + timeout

    while time.time() < deadline:
        response = subscriber.pull(
            request={
                "subscription": subscription_path,
                "max_messages": 10
            }
        )

        ack_ids = []

        for msg in response.received_messages:
            payload = json.loads(msg.message.data.decode("utf-8"))
            pred = payload.get("prediction", {})

            if pred.get("flower_id") == flower_id:
                ack_ids.append(msg.ack_id)

                subscriber.acknowledge(
                    request={
                        "subscription": subscription_path,
                        "ack_ids": ack_ids
                    }
                )
                return payload

            ack_ids.append(msg.ack_id)

        if ack_ids:
            subscriber.acknowledge(
                request={
                    "subscription": subscription_path,
                    "ack_ids": ack_ids
                }
            )

        time.sleep(1)

    return {"error": "Prediction timeout"}
