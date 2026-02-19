import os
import mlflow
import dagshub
import yaml
import pickle
import pandas as pd
from sklearn.metrics import  accuracy_score
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

dagshub.init(repo_owner='saadtariq-ds', repo_name='diabetes-classification', mlflow=True)

## Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(model_path, data_path):
    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load the test data
    data = pd.read_csv(data_path)
    
    # Split features and target
    X_test = data.drop(columns=["Outcome"])
    y_test = data["Outcome"]

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Log the accuracy to MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    print("Test accuracy logged to MLflow")


if __name__ == "__main__":
    evaluate(
        model_path=params["model_dir"], 
        data_path=params["data_dir"]
    )