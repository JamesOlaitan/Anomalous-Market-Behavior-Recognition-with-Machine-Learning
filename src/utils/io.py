"""Input/Output utilities."""
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import torch


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.

    Args:
        file_path: Input file path

    Returns:
        Loaded dictionary
    """
    with open(file_path, "r") as f:
        return json.load(f)


def save_pickle(obj: Any, file_path: str) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str) -> Any:
    """
    Load object from pickle file.

    Args:
        file_path: Input file path

    Returns:
        Loaded object
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_model(model: torch.nn.Module, file_path: str) -> None:
    """
    Save PyTorch model.

    Args:
        model: PyTorch model
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), file_path)


def load_model(model: torch.nn.Module, file_path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load PyTorch model.

    Args:
        model: PyTorch model instance
        file_path: Input file path
        device: Device to load model on

    Returns:
        Model with loaded weights
    """
    model.load_state_dict(torch.load(file_path, map_location=device))
    return model
