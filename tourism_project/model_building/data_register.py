
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "tourism.csv"
HF_DATASET_REPO = "bhumitps/tourism_dataset"   # Hugging Face dataset repo
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in environment variables")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully!")
print("Shape:", df.shape)

# -----------------------------
# CONNECT TO HUGGING FACE
# -----------------------------
api = HfApi(token=HF_TOKEN)

# -----------------------------
# CREATE DATASET REPO (IF NOT EXISTS)
# -----------------------------
try:
    api.repo_info(HF_DATASET_REPO, repo_type="dataset")
    print(f"Dataset repo '{HF_DATASET_REPO}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{HF_DATASET_REPO}' not found. Creating it...")
    create_repo(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        private=False
    )

# -----------------------------
# UPLOAD DATASET FILE
# -----------------------------
api.upload_file(
    path_or_fileobj=DATASET_PATH,
    path_in_repo="tourism.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
)

print("Dataset successfully registered to Hugging Face!")
