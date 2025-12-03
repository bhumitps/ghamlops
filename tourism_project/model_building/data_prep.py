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
HF_DATASET_REPO = "bhumitps/tourism_dataset"   # dataset repo on HF
DATASET_FILENAME = "tourism.csv"

# Target column: whether customer purchased the Wellness Tourism Package
TARGET_COL = "ProdTaken"   # <- most common name for this problem

PROCESSED_DIR = "tourism_project/processed_data"
os.makedirs(PROCESSED_DIR, exist_ok=True)


if HF_TOKEN is None:
    raise ValueError(
        "HF_TOKEN not found in environment variables. "
        "Set HF_TOKEN in your local env or GitHub Actions secrets."
    )


def load_dataset_from_hf():
    """Download tourism.csv from HF dataset repo and load into a DataFrame."""
    local_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=DATASET_FILENAME,
        token=HF_TOKEN,
    )
    df = pd.read_csv(local_path)
    print(f"Loaded dataset from HF: {local_path}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates and obvious ID columns if present."""
    df = df.drop_duplicates().reset_index(drop=True)

    # Drop obvious ID-like columns if they exist
    id_cols = ["CustomerID", "Customer_ID", "ID", "id"]
    to_drop = [c for c in id_cols if c in df.columns]
    if to_drop:
        print("Dropping ID columns:", to_drop)
        df = df.drop(columns=to_drop)

    # Simple missing-value handling (you can refine this per feature later)
    # For now, drop rows with NA in target, and impute others simply.
    if TARGET_COL in df.columns:
        df = df[~df[TARGET_COL].isna()].copy()

    # Numeric: fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Non-numeric: fill with mode
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categoricals(df: pd.DataFrame, target_col: str):
    """
    Label-encode all object/category columns (including target if needed).
    Returns encoded feature dataframe X and encoded target series y.
    """
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(
            f"TARGET_COL '{target_col}' not found in columns: {df.columns.tolist()}"
        )

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode target if it is not numeric
    if not np.issubdtype(y.dtype, np.number):
        print(f"Encoding target column '{target_col}' with LabelEncoder.")
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    else:
        y = y.values

    # Encode categorical features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        print("Encoding categorical feature columns:", cat_cols)
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    return X, pd.Series(y, name=target_col)


def main():
    # 1. Load dataset
    df = load_dataset_from_hf()

    # 2. Basic cleaning
    df = basic_cleaning(df)

    # 3. Encoding
    X, y = encode_categoricals(df, TARGET_COL)

    print("Final feature columns:", X.columns.tolist())
    print("X shape:", X.shape, "y shape:", y.shape)

    # 4. Train/test split (stratified for classification)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # helps preserve class balance in splits
    )

    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes:", X_test.shape, y_test.shape)

    # 5. Save processed splits
    xtrain_path = os.path.join(PROCESSED_DIR, "xtrain.csv")
    xtest_path  = os.path.join(PROCESSED_DIR, "xtest.csv")
    ytrain_path = os.path.join(PROCESSED_DIR, "ytrain.csv")
    ytest_path  = os.path.join(PROCESSED_DIR, "ytest.csv")

    X_train.to_csv(xtrain_path, index=False)
    X_test.to_csv(xtest_path, index=False)
    y_train.to_csv(ytrain_path, index=False)
    y_test.to_csv(ytest_path, index=False)

    print(f"Saved X_train to: {xtrain_path}")
    print(f"Saved X_test  to: {xtest_path}")
    print(f"Saved y_train to: {ytrain_path}")
    print(f"Saved y_test  to: {ytest_path}")
    print(" Data preparation complete.")


if __name__ == "__main__":
    main()
