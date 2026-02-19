import os
import yaml
import pickle
import mlflow
import pandas as pd
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mlflow.models import infer_signature
from urllib.parse import urlparse

dagshub.init(repo_owner='saadtariq-ds', repo_name='diabetes-classification', mlflow=True)

## Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def hyperparameter_tuning(X_train, y_train, param_grid):
    classifier = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=classifier, 
        param_grid=param_grid, 
        cv=3, 
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def train(data_path, model_output_path, random_state):
    # Load preprocessed data
    data = pd.read_csv(data_path)
    
    # Split features and target
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Start MLflow run
    with mlflow.start_run():

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        # MLflow Signature
        signature = infer_signature(X_train, y_train)

        # Hyperparameter Tuning
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        
        print(f"Best Hyperparameters: {grid_search.best_params_}")

        # Train the model with the best hyperparameters
        best_classifier = grid_search.best_estimator_
        best_classifier.fit(X_train, y_train)

        # Evaluate the model
        y_pred = best_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        ## Log parameters, metrics, and model to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_n_estimators", grid_search.best_params_["n_estimators"])
        mlflow.log_param("best_max_depth", grid_search.best_params_["max_depth"])
        mlflow.log_param("best_min_samples_split", grid_search.best_params_["min_samples_split"])
        mlflow.log_param("best_min_samples_leaf", grid_search.best_params_["min_samples_leaf"])

        mlflow.log_text(str(matrix), "confusion_matrix.txt")
        mlflow.log_text(report, "classification_report.txt")

        tracking_url_type_store = urlparse(
            mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_classifier, "model", registered_model_name="DiabetesClassifier")
        else:
            mlflow.sklearn.log_model(best_classifier, "model", signature=signature)

        # Create the output directory if it doesn't exist to save the model
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        file_name = model_output_path
        pickle.dump(best_classifier, open(file_name, "wb"))
        print(f"Model saved to {file_name}")


if __name__ == "__main__":
    train(
        data_path=params["data_dir"],
        model_output_path=params["model_dir"],
        random_state=params["random_state"]
    )