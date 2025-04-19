import os

import kagglehub


def download_data():
    """
    Downloads the dataset and moves it to the project directory.
    """
    path = kagglehub.dataset_download("sttaseen/wlasl2000-resized")
    print(f"Dataset downloaded to: {path}")

    # Move to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)
    os.rename(path, os.path.join(project_dir, "data", "raw"))


if __name__ == "__main__":
    download_data()
