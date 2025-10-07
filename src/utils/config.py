"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure all necessary directories exist.

    Args:
        config: Configuration dictionary
    """
    paths = config.get("paths", {})
    for key, path in paths.items():
        if key.endswith("_dir"):
            os.makedirs(path, exist_ok=True)


def get_device(config: Dict[str, Any]) -> str:
    """
    Determine the compute device based on config and availability.

    Args:
        config: Configuration dictionary

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    import torch

    device_config = config.get("device", "auto")

    if device_config == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    else:
        return device_config
