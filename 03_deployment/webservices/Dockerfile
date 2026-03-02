# write a dockerfile to containerize the predict.py script. The dockerfile should use python:3.9-slim as the base image, install the necessary dependencies, copy the predict.py script into the container, and set the entrypoint to run the predict.py script.
# Dockerfile
FROM python:3.9-slim    
WORKDIR /app
# predict is not in the same directory as the Dockerfile, so we need to specify the path to it
# update the code to copy the predict.py script from the correct location
COPY predict.py .
# copy the scaler.joblib file from the correct location
COPY scaler.joblib .
RUN pip install mlflow joblib numpy scikit-learn fastapi uvicorn
# export mlflow tracking uri and registry uri as environment variables
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5001
ENV MLFLOW_REGISTRY_URI=http://host.docker.internal:5001
# mount the mlruns directory as a volume to access the model artifacts
VOLUME "/Users/iashahba/Documents/ARTIFIN/FS26/SW2/Data and Code/mlruns:/mlruns"
EXPOSE 8000
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]

# To build the docker image, run the following command in the terminal:
# docker build -t predict-service .
# To run the docker container, run the following command in the terminal:
# docker run -p 8000:8000 predict-service


