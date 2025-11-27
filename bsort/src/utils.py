"""Utility functions for bsort."""

import os
import zipfile
from pathlib import Path
from typing import Optional

import gdown


def download_and_extract_dataset(url: str, output_path: str) -> str:
    """Download and extract dataset from Google Drive.

    Args:
        url: Google Drive URL
        output_path: Output directory path

    Returns:
        Path to extracted dataset directory

    Raises:
        Exception: If download or extraction fails
    """
    output_path = Path(output_path)
    zip_path = output_path / "dataset.zip"

    # Create directory if not exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Download
    gdown.download(url, str(zip_path), fuzzy=True)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_path)

    # Find extracted directory
    extracted_dirs = [d for d in output_path.iterdir() if d.is_dir()]
    if extracted_dirs:
        return str(extracted_dirs[0])
    else:
        return str(output_path)


def setup_directories() -> None:
    """Create necessary directories for training."""
    directories = ["data", "runs"]

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
