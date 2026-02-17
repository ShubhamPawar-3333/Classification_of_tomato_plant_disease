"""
Common utility functions for the Tomato Disease Advisory System.
"""
import os
import yaml
import json
from pathlib import Path
from typing import Any
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read a YAML file and return its contents as a ConfigBox.
    
    Args:
        path_to_yaml: Path to the YAML file
        
    Returns:
        ConfigBox: Contents of the YAML file as a ConfigBox object
        
    Raises:
        ValueError: If the YAML file is empty
        Exception: If there's an error reading the file
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError(f"YAML file is empty: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"Error parsing YAML file: {path_to_yaml}")
    except Exception as e:
        raise e


@ensure_annotations
def read_json(path_to_json: Path) -> dict:
    """
    Read a JSON file and return its contents as a dictionary.
    
    Args:
        path_to_json: Path to the JSON file
        
    Returns:
        dict: Contents of the JSON file
    """
    with open(path_to_json) as json_file:
        content = json.load(json_file)
    return content


@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """
    Save data to a JSON file.
    
    Args:
        path: Path where the JSON file will be saved
        data: Data to save
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# @ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True) -> None:
    """
    Create a list of directories.
    
    Args:
        path_to_directories: List of paths to directories to create
        verbose: Whether to log directory creation
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"Created directory: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get the size of a file in KB.
    
    Args:
        path: Path to the file
        
    Returns:
        str: Size of the file in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"
