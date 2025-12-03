import os
import pandas as pd
import numpy as np

from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# CONFIG
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = "bhumitps/tourism_dataset"
DATASET_FILENAME = "tourism.csv"

TARGET_COL = "ProdTaken"

# IMPORTANT:
# This script is run with working-directory: tourism_project (in YAML),
# so all paths here are relative to tourism_project/.
PROCESSED_DIR = "data"   # <--- ONLY "data", not "tourism_project/data"
os.makedirs(PROCESSED_DIR, exist_ok=True)

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing or empty. Set it as a GitHub Actions secret.")


def main():
    # 1. Download dataset from Hugging Face
    local_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=DATASET_FILENAME,
        token=HF_TOKEN,
    )

    df = pd.read_csv(local_path)
    print("Dataset downloaded from HF")
    print("Shape:", df.shape)

    # 2. Basic cleaning
    df = df.drop_duplicates().reset_index(drop=True)

    # 3. Split X, y
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset columns: {df.columns.tolist()}")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # 4. Encode categorical features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        print("Encoding categorical columns:", cat_cols)
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    print("Final X shape:", X.shape, "y shape:", y.shape)

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 6. Save processed files under tourism_project/data/
    xtrain_path = os.path.join(PROCESSED_DIR, "xtrain.csv")
    xtest_path  = os.path.join(PROCESSED_DIR, "xtest.csv")
    ytrain_path = os.path.join(PROCESSED_DIR, "ytrain.csv")
    ytest_path  = os.path.join(PROCESSED_DIR, "ytest.csv")

    X_train.to_csv(xtrain_path, index=False)
    X_test.to_csv(xtest_path, index=False)
    y_train.to_csv(ytrain_path, index=False)
    y_test.to_csv(ytest_path, index=False)

    print("Saved X_train to:", xtrain_path)
    print("Saved X_test  to:", xtest_path)
    print("Saved y_train to:", ytrain_path)
    print("Saved y_test  to:", ytest_path)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
