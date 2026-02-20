#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn.datasets import load_iris
import pandas as pd


# Load the iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows
df.head()


# In[19]:


# is my data balanced or imbalanced?
df.target.value_counts()


# In[20]:


# Check for missing values in each column
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Total missing values in the dataset
total_missing = df.isnull().sum().sum()
print(f"\nTotal missing values: {total_missing}")

# Alternatively, get a summary of data types and non-null counts
df.info()


# In[21]:


df.describe()


# In[22]:


df.columns


# In[23]:


from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# First split: train and temp (temp will be split into validation and test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Second split: validation and test from temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Check sizes
print(f"Train set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# In[24]:


import mlflow
import mlflow.sklearn
import shutil

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os 

# Remove corrupted mlruns directory if it exists
#if os.path.exists("./mlruns"):
    #shutil.rmtree("./mlruns")

# Create mlruns directory
os.makedirs("./mlruns", exist_ok=True)
# Set up local tracking URI (creates a 'mlruns' folder in your current directory)
mlflow.set_tracking_uri("file:./mlruns")

# Create or set an experiment
mlflow.set_experiment("Iris Baseline Experiment")

solver = ["lbfgs", "newton-cg", "sag", "saga"]

for s in solver:
    # Start an MLflow run for each solver
    with mlflow.start_run():
        # Initialize and train the model
        model = LogisticRegression(random_state=42, max_iter=200, solver=s)
        model.fit(X_train, y_train)

        # Predict on validation set
        y_val_pred = model.predict(X_val)
        #print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
        #print("Validation Classification Report:")
        #print(classification_report(y_val, y_val_pred))

        # Predict on test set
        y_test_pred = model.predict(X_test)
        #print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
        #print("Test Classification Report:")
        #print(classification_report(y_test, y_test_pred))
        
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("solver", s)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", 200)
        
        # Log metrics
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("test_precision_macro", classification_report(y_test, y_test_pred, output_dict=True)['macro avg']['precision'])
        mlflow.log_metric("test_recall_macro", classification_report(y_test, y_test_pred, output_dict=True)['macro avg']['recall'])
        mlflow.log_metric("test_f1_macro", classification_report(y_test, y_test_pred, output_dict=True)['macro avg']['f1-score'])
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        # Get the run ID for registration
        run_id = mlflow.active_run().info.run_id

    # Register the model outside the run context
    mlflow.register_model(f"runs:/{run_id}/model", f"iris_baseline_{s}")


# In[25]:


# After the loop, find the best run based on test_accuracy
best_runs = mlflow.search_runs(experiment_names=["Iris Baseline Experiment"], order_by=["metrics.test_accuracy DESC"])
best_run = best_runs.iloc[0]
best_run_id = best_run['run_id']
best_solver = best_run['params.solver']
best_accuracy = best_run['metrics.test_accuracy']

print(f"\nBest Model: {best_solver}")
print(f"Best Test Accuracy: {best_accuracy}")

# Register the best model as the baseline
mlflow.register_model(f"runs:/{best_run_id}/model", "iris_baseline")

print("Best model registered as 'iris_baseline'")


# In[26]:


from mlflow.tracking import MlflowClient

client = MlflowClient()

model_version = 1
new_stage = "Staging"
client.transition_model_version_stage(
    name="iris_baseline",
    version=model_version,
    stage=new_stage,
    archive_existing_versions=False
)

print(f"Model 'iris_baseline' version {model_version} transitioned to {new_stage}")


# In[29]:


from datetime import datetime   
date = datetime.today().strftime("%Y-%m-%d")


# Update the model version description
client.update_model_version(
    name="iris_baseline",
    version=model_version,
    description=f"Best model from Iris Baseline Experiment with solver {best_solver} and test accuracy {best_accuracy:.4f} as of {date}"
)


# In[27]:


from mlflow.tracking import MlflowClient
from datetime import datetime   

client = MlflowClient()
date = datetime.today().strftime("%Y-%m-%d")

# Use version 1 (the first registered version)
model_version = 1
new_stage = "Staging"

client.transition_model_version_stage(
    name="iris_baseline",
    version=model_version,
    stage=new_stage,
    archive_existing_versions=False
)

print(f"Model 'iris_baseline' version {model_version} transitioned to {new_stage}")

# Update the model version description
client.update_model_version(
    name="iris_baseline",
    version=model_version,
    description=f"Best model from Iris Baseline Experiment with solver {best_solver} and test accuracy {best_accuracy:.4f} as of {date}"
)


# In[28]:


import mlflow.pyfunc

# Load the best model from the model registry
model_uri = "models:/iris_baseline/Staging"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Make predictions on test set
predictions = loaded_model.predict(X_test)

# Display results
print("Predictions on test set:")
print(predictions[:10])  # First 10 predictions

# Evaluate
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, predictions)
print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

