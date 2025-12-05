import os
import pandas as pd
import numpy as np

from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = "bhumitps/tourism_dataset"
DATASET_FILENAME = "tourism.csv"
TARGET_COL = "ProdTaken"

# We run with working-directory = tourism_project, so this is tourism_project/data
PROCESSED_DIR = "data"
os.makedirs(PROCESSED_DIR, exist_ok=True)

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing or empty. Set it as a GitHub Actions secret.")


def main():
    # 1. Download dataset from HF
    local_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=DATASET_FILENAME,
        token=HF_TOKEN,
    )

    df = pd.read_csv(local_path)
    print("Dataset downloaded from HF:", local_path)
    print("Original shape:", df.shape)

    # 2. Basic cleaning
    df = df.drop_duplicates().reset_index(drop=True)

    # 3. Drop ID / index-like columns that should not be used as features
    cols_to_drop = ["CustomerID"]  # known ID column
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    cols_to_drop += unnamed_cols

    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if cols_to_drop:
        print("Dropping non-feature columns:", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    # 4. Split features / target
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    print("Shape after dropping cols & target:")
    print("X shape:", X.shape, "y shape:", y.shape)

    # ❗ No encoding here – keep raw strings.
    #   Categorical handling is moved into the sklearn Pipeline in train.py.

    # 5. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Save to CSVs for training job
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


if __name__ == "__main__":
    main()
