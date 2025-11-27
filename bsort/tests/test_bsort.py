"""Unit tests for bsort."""

import pytest
from bsort.config import Config
from bsort.utils import validate_config


def test_config_loading():
    """Test configuration loading from YAML."""
    config = Config(
        dataset_url="test_url",
        epochs=10,
        batch_size=8,
        learning_rate=0.001,
    )

    assert config.dataset_url == "test_url"
    assert config.epochs == 10
    assert config.batch_size == 8
    assert config.learning_rate == 0.001


def test_config_validation():
    """Test configuration validation."""
    valid_config = {
        "dataset_url": "test",
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.001,
    }

    invalid_config = {"epochs": 10, "batch_size": 8}

    assert validate_config(valid_config) == True
    assert validate_config(invalid_config) == False


def test_config_to_dict():
    """Test config to dictionary conversion."""
    config = Config(
        dataset_url="test_url",
        epochs=10,
        batch_size=8,
        learning_rate=0.001,
    )

    config_dict = config.to_dict()
    assert config_dict["dataset_url"] == "test_url"
    assert config_dict["epochs"] == 10
    assert config_dict["batch_size"] == 8
    assert config_dict["learning_rate"] == 0.001
