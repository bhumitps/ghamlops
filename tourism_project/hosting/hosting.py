import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# -----------------------------
# CONFIG
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_REPO_ID = "bhumitps/amlops"          # Hugging Face Space (Docker)
LOCAL_APP_FOLDER = "tourism_project/deployment"  # Folder with Dockerfile, app.py, requirements.txt

if HF_TOKEN is None:
    raise ValueError(
        "HF_TOKEN not found in environment variables. "
        "Set HF_TOKEN in your local env or GitHub Actions secrets."
    )


def ensure_space_repo(api: HfApi, repo_id: str):
    """
    Ensure the Space repo exists. If not, create it as a Docker Space.
    """
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"Space '{repo_id}' already exists. Space SDK: {getattr(info, 'space_sdk', 'unknown')}")
    except (RepositoryNotFoundError, HfHubHTTPError):
        print(f"Space '{repo_id}' not found. Creating a new Docker Space...")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",  # important: we are using Dockerfile-based Space
            private=False,
            exist_ok=True,
        )
        print(f"Space '{repo_id}' created.")


def upload_app_folder(api: HfApi, local_folder: str, repo_id: str):
    """
    Upload the local deployment folder to the Space.
    """
    if not os.path.isdir(local_folder):
        raise FileNotFoundError(
            f"Local app folder '{local_folder}' not found. "
            "Make sure the path is correct and files are generated."
        )

    print(f"Uploading contents of '{local_folder}' to Space '{repo_id}'...")
    api.upload_folder(
        folder_path=local_folder,
        repo_id=repo_id,
        repo_type="space",
        path_in_repo=".",   # upload into root of the Space
    )
    print("Deployment files uploaded successfully.")


def main():
    api = HfApi(token=HF_TOKEN)

    # 1. Ensure the Space repo exists
    ensure_space_repo(api, SPACE_REPO_ID)

    # 2. Upload deployment artifacts (Dockerfile, app.py, requirements.txt, etc.)
    upload_app_folder(api, LOCAL_APP_FOLDER, SPACE_REPO_ID)


if __name__ == "__main__":
    main()
