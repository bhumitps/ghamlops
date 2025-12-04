import os
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import xgboost as xgb

import mlflow
import mlflow.xgboost

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


# -----------------------------
# CONFIG
# -----------------------------
# This script is run with working-directory = tourism_project
PROCESSED_DIR = "data"   #

XTRAIN_PATH = os.path.join(PROCESSED_DIR, "xtrain.csv")
XTEST_PATH  = os.path.join(PROCESSED_DIR, "xtest.csv")
YTRAIN_PATH = os.path.join(PROCESSED_DIR, "ytrain.csv")
YTEST_PATH  = os.path.join(PROCESSED_DIR, "ytest.csv")

TARGET_COL = "ProdTaken"

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REPO_ID = "bhumitps/tourism_model"

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tourism-mlops-experiment")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is missing or empty. Set it as a GitHub Actions secret."
    )


def load_processed_data():
    print("CWD:", os.getcwd())
    print("Looking for:", os.path.abspath(XTRAIN_PATH))

    if not os.path.exists(XTRAIN_PATH):
        raise FileNotFoundError(f"xtrain not found at: {os.path.abspath(XTRAIN_PATH)}")

    X_train = pd.read_csv(XTRAIN_PATH)
    X_test  = pd.read_csv(XTEST_PATH)
    y_train = pd.read_csv(YTRAIN_PATH).squeeze()
    y_test  = pd.read_csv(YTEST_PATH).squeeze()

    print("Loaded processed data:")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("y_train:", y_train.shape, "y_test:", y_test.shape)

    return X_train, X_test, y_train, y_test


def build_model(class_weight=None):
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=class_weight if class_weight is not None else 1.0,
    )

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", xgb_clf),
    ])

    return pipeline


def get_class_weight(y_train):
    value_counts = pd.Series(y_train).value_counts()
    if 0 in value_counts and 1 in value_counts:
        class_weight = value_counts[0] / value_counts[1]
        print("Class counts:", value_counts.to_dict())
        print("Computed scale_pos_weight:", class_weight)
        return class_weight
    else:
        print("Could not find both classes 0 and 1, using class_weight = 1.0")
        return 1.0


def train_and_log():
    X_train, X_test, y_train, y_test = load_processed_data()
    class_weight = get_class_weight(y_train)
    model_pipeline = build_model(class_weight=class_weight)

    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("scale_pos_weight", class_weight)

        model_pipeline.fit(X_train, y_train)

        y_train_pred = model_pipeline.predict(X_train)
        y_test_pred  = model_pipeline.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc  = accuracy_score(y_test, y_test_pred)

        print("\nTrain accuracy:", train_acc)
        print("Test accuracy:", test_acc)

        print("\nClassification report (test):")
        print(classification_report(y_test, y_test_pred))

        cm = confusion_matrix(y_test, y_test_pred)
        print("Confusion matrix (test):\n", cm)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_tp", int(cm[1, 1]))
        mlflow.log_metric("test_fp", int(cm[0, 1]))
        mlflow.log_metric("test_tn", int(cm[0, 0]))
        mlflow.log_metric("test_fn", int(cm[1, 0]))

        mlflow.xgboost.log_model(
            xgb_model=model_pipeline.named_steps["model"],
            artifact_path="xgb_model_raw"
        )

        model_path = "best_tourism_model_v1.joblib"
        joblib.dump(model_pipeline, model_path)
        print(f"\nSaved full pipeline to {model_path}")

        mlflow.log_artifact(model_path, artifact_path="model")

        upload_model_to_hf(model_path)


def upload_model_to_hf(model_path: str):
    print("\nUploading model to Hugging Face Hub...")
    api = HfApi(token=HF_TOKEN)

    try:
        api.repo_info(MODEL_REPO_ID, repo_type="model")
        print(f"Model repo '{MODEL_REPO_ID}' already exists.")
    except (RepositoryNotFoundError, HfHubHTTPError):
        print(f"Model repo '{MODEL_REPO_ID}' not found. Creating...")
        create_repo(
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            private=False
        )

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=MODEL_REPO_ID,
        repo_type="model",
    )

    print("Model uploaded to Hugging Face model repo:", MODEL_REPO_ID)


if __name__ == "__main__":
    train_and_log()
