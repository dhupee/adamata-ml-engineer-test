"""Configuration management for bsort."""
from typing import Optional, Dict, Any
import yaml
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for model training and inference.
    
    Attributes:
        dataset_url: URL to download dataset from
        dataset_path: Local path for dataset storage
        image_size: Image size for training/inference
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        model_name: Pretrained model name
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases entity/username
        export_formats: List of formats to export model to
    """
    
    # Data configuration
    dataset_url: str
    dataset_path: str = "data"
    image_size: int = 320
    
    # Training configuration
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    model_name: str = "yolo11n.pt"
    
    # W&B configuration
    wandb_project: str = "adamata-bsort"
    wandb_entity: Optional[str] = None
    
    # Export configuration
    export_formats: list = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.export_formats is None:
            self.export_formats = ["onnx", "torchscript"]
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            Config instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            'dataset_url': self.dataset_url,
            'dataset_path': self.dataset_path,
            'image_size': self.image_size,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'model_name': self.model_name,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity,
            'export_formats': self.export_formats
        }