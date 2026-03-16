# Dockerfile
FROM python:3.9-slim    
WORKDIR /app
COPY predict.py .
COPY scaler.joblib .
RUN pip install mlflow joblib numpy scikit-learn fastapi uvicorn
# export mlflow tracking uri and registry uri as environment variables
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5001
ENV MLFLOW_REGISTRY_URI=http://host.docker.internal:5001
EXPOSE 8000
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]

# To build the docker image, run the following command in the terminal:
# docker build -t predict-service .
# To run the docker container, run the following command in the terminal:
# docker run -p 8000:8000 predict-service


