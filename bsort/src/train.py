"""Training module for bsort."""

from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO

from src.config import Config
from src.utils import download_and_extract_dataset, setup_directories


def train_model(config: Config) -> Dict[str, Any]:
    """Train the YOLO model with given configuration.

    Args:
        config: Configuration object

    Returns:
        Dictionary containing training results

    Raises:
        FileNotFoundError: If data.yaml is not found in dataset
        Exception: For any training errors
    """
    # Setup directories
    setup_directories()

    try:
        # Download dataset if not exists
        dataset_dir = download_and_extract_dataset(config.dataset_url, config.dataset_path)

        data_yaml_path = Path(dataset_dir) / "data.yaml"
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

        # Load model
        model = YOLO(config.model_name)

        # Train model
        results = model.train(
            data=str(data_yaml_path),
            epochs=config.epochs,
            imgsz=config.image_size,
            batch=config.batch_size,
            lr0=config.learning_rate,
            project="bsort-training",
            name="run",
        )

        # Export model in different format
        export_paths = {}
        for fmt in config.export_formats:
            try:
                export_path = model.export(format=fmt)
                export_paths[fmt] = export_path
                print(f"✓ Model exported to {fmt.upper()} format: {export_path}")
            except Exception as e:
                print(f"✗ Failed to export to {fmt}: {e}")

        # Save model
        model_path = config.model_path
        model.save(model_path)
        print(f"✓ Model saved as PyTorch: {model_path}")

        return {"success": True, "model_path": model_path, "export_paths": export_paths, "results": results}

    except Exception as e:
        raise e
