"""
Configuration management utilities for UAV SLB project.



This module provides functions to load, validate, and manage configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """
    Load a YAML configuration file using OmegaConf.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        OmegaConf DictConfig object with configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        # Load with OmegaConf for variable interpolation support
        config = OmegaConf.load(config_path)
        return config
    except Exception as e:
        raise yaml.YAMLError(f"Error loading config from {config_path}: {e}")


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load a YAML file using standard PyYAML (simpler, no interpolation).
    
    Args:
        yaml_path: Path to the YAML file
        
    Returns:
        Dictionary with configuration
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """
    Save a configuration to a YAML file.
    
    Args:
        config: OmegaConf DictConfig or dict to save
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert OmegaConf to dict if needed
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multiple configurations with priority given to later configs.
    
    Args:
        *configs: Variable number of DictConfig objects to merge
        
    Returns:
        Merged configuration
    """
    if not configs:
        return OmegaConf.create({})
    
    merged = configs[0]
    for config in configs[1:]:
        merged = OmegaConf.merge(merged, config)
    
    return merged


def validate_data_config(config: DictConfig) -> None:
    """
    Validate data configuration has required fields and sensible values.
    
    Args:
        config: Data configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required top-level keys
    required_keys = ['data', 'task', 'preprocessing', 'dataloader']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
    
    # Validate task type
    valid_task_types = ['classification', 'regression']
    if config.task.type not in valid_task_types:
        raise ValueError(
            f"Invalid task type: {config.task.type}. "
            f"Must be one of {valid_task_types}"
        )
    
    # Validate splits sum to 1.0
    if 'splits' in config:
        total = (config.splits.train_ratio + 
                 config.splits.val_ratio + 
                 config.splits.test_ratio)
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Data splits must sum to 1.0, got {total}"
            )
    
    # Validate image size
    if len(config.preprocessing.image_size) != 2:
        raise ValueError(
            "preprocessing.image_size must be [height, width], "
            f"got {config.preprocessing.image_size}"
        )
    
    # Validate normalization stats
    if len(config.preprocessing.normalize.mean) != 3:
        raise ValueError(
            "preprocessing.normalize.mean must have 3 values (RGB)"
        )
    if len(config.preprocessing.normalize.std) != 3:
        raise ValueError(
            "preprocessing.normalize.std must have 3 values (RGB)"
        )


def resolve_paths(config: DictConfig, relative_to: Optional[str] = None) -> DictConfig:
    """
    Resolve all relative paths in configuration to absolute paths.
    
    Args:
        config: Configuration with potentially relative paths
        relative_to: Base directory for relative paths. If None, uses current working directory
        
    Returns:
        Configuration with resolved absolute paths
    """
    if relative_to is None:
        relative_to = os.getcwd()
    
    base_path = Path(relative_to)
    
    # Deep copy to avoid modifying original
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    
    # Resolve data paths
    if 'data' in config:
        for key, value in config.data.items():
            if isinstance(value, str) and ('dir' in key or 'path' in key):
                path = Path(value)
                if not path.is_absolute():
                    config.data[key] = str((base_path / path).resolve())
    
    return config


def get_project_root() -> Path:
    """
    Get the project root directory (where environment.yml is located).
    
    Returns:
        Path to project root
    """
    current = Path(__file__).resolve()
    
    # Navigate up from src/utils/config.py to project root
    # Assuming structure: project_root/src/utils/config.py
    project_root = current.parent.parent.parent
    
    # Verify this is indeed the project root
    if not (project_root / 'environment.yml').exists():
        # Fallback: search upward for environment.yml
        for parent in current.parents:
            if (parent / 'environment.yml').exists():
                return parent
        raise FileNotFoundError(
            "Could not find project root (looking for environment.yml)"
        )
    
    return project_root


def get_config_dir() -> Path:
    """
    Get the configs directory path.
    
    Returns:
        Path to configs directory
    """
    return get_project_root() / 'configs'


def print_config(config: DictConfig, resolve: bool = True) -> None:
    """
    Pretty print configuration for debugging.
    
    Args:
        config: Configuration to print
        resolve: Whether to resolve interpolations
    """
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(config, resolve=resolve))
    print("=" * 80)


# Example usage
if __name__ == "__main__":
    # Load data config
    config_dir = get_config_dir()
    data_config = load_config(config_dir / "data_config.yaml")
    
    # Validate
    validate_data_config(data_config)
    
    # Print
    print_config(data_config)
