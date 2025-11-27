"""Utility functions for bsort."""

import os
import zipfile
from pathlib import Path

import requests


def download_and_extract_dataset(url: str, output_path: str) -> str:
    """Download and extract dataset from Google Drive.

    Args:
        url: Roboflow Dataset URL
        output_path: Output directory path

    Returns:
        Path to extracted dataset directory

    Raises:
        Exception: If download or extraction fails
    """

    zip_path = output_path + "/" + "dataset.zip"

    # Create directory if not exists
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True, exist_ok=True)

    try:
        # Send GET request with stream=True to handle large files
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Write the content to a file
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_path)

    # Find extracted directory
    return output_path


def setup_directories() -> None:
    """Create necessary directories for training."""
    directories = ["data"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def validate_config(config_dict: dict) -> bool:
    """Validate configuration dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "dataset_url",
        "epochs",
        "batch_size",
        "learning_rate",
    ]

    for field in required_fields:
        if field not in config_dict:
            return False

    return True
