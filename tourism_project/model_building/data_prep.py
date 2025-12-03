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

PROCESSED_DIR = "tourism_project/data"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -----------------------------
# LOAD DATASET FROM HF
# -----------------------------
local_path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    filename=DATASET_FILENAME,
    token=HF_TOKEN,
)

df = pd.read_csv(local_path)
print("Dataset downloaded from HF")
print("Shape:", df.shape)

# -----------------------------
# BASIC CLEANING
# -----------------------------
df = df.drop_duplicates().reset_index(drop=True)

# -----------------------------
# ENCODING
# -----------------------------
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

for col in X.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# SAVE PROCESSED FILES MATCHING YOUR REPO STRUCTURE
# -----------------------------
X_train.to_csv("tourism_project/data/xtrain.csv", index=False)
X_test.to_csv("tourism_project/data/xtest.csv", index=False)
y_train.to_csv("tourism_project/data/ytrain.csv", index=False)
y_test.to_csv("tourism_project/data/ytest.csv", index=False)

print("Data preparation complete and files saved.")
