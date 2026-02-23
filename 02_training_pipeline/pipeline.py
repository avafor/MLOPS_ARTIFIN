#!/usr/bin/env python
# coding: utf-8


from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os 
from sklearn.preprocessing import StandardScaler


# Load the iris dataset
def load_data():
    iris = load_iris()

    # Create a DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df




def check_data_quality(df):
    # Check for missing values in each column
    missing_values = df.isnull().sum()
    print("Missing values in each column:")
    print(missing_values)   
    return missing_values



def test_train_val_test_split(df):


    # Define features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # First split: train and temp (temp will be split into validation and test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    # Second split: validation and test from temp
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardizer (X_train):
    # Fit scaler on training data (do this once after training)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler 


def train_and_log_models(X_train, X_val, X_test, y_train, y_val, y_test):

    # Create mlruns directory
    os.makedirs("./mlruns", exist_ok=True)
    # Set up local tracking URI (creates a 'mlruns' folder in your current directory)
    mlflow.set_tracking_uri("file:./mlruns")

    # Create or set an experiment
    mlflow.set_experiment("Iris Baseline Experiment")

    # Start an MLflow run for each solver
    with mlflow.start_run():
        # Initialize and train the model
        model = LogisticRegression(random_state=42, max_iter=200, solver='saga')
        model.fit(X_train, y_train)

        # Predict on validation set
        y_val_pred = model.predict(X_val)
        # Predict on test set
        y_test_pred = model.predict(X_test)
        
        
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("solver", s)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", 200)
        
        # Log metrics
        mlflow.log_metric("val_accuracy", accuracy_score(y_val, y_val_pred))
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("test_precision_macro", classification_report(y_test, y_test_pred, output_dict=True)['macro avg']['precision'])
        mlflow.log_metric("test_recall_macro", classification_report(y_test, y_test_pred, output_dict=True)['macro avg']['recall'])
        mlflow.log_metric("test_f1_macro", classification_report(y_test, y_test_pred, output_dict=True)['macro avg']['f1-score'])
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        # Get the run ID for registration
        run_id = mlflow.active_run().info.run_id

    # Register the model outside the run context
    mlflow.register_model(f"runs:/{run_id}/model", f"iris_model")
    return run_id

if __name__ == "__main__":
    df = load_data()
    missing_values = check_data_quality(df)
    X_train, X_val, X_test, y_train, y_val, y_test = test_train_val_test_split(df)
    scaler = standardizer (X_train)
    run_id = train_and_log_models(X_train, X_val, X_test, y_train, y_val, y_test)

    