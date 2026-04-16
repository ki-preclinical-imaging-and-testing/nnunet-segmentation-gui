# src/utils/helpers.py
"""Helper functions"""
import os
from pathlib import Path
import yaml
from typing import Dict, Optional


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML"""
    config_path = Path(config_path)
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    return {}


def get_models_dir(config: Dict) -> str:
    """Get models directory from config or environment"""
    # Check environment variable first
    if 'NNUNET_MODELS' in os.environ:
        return os.environ['NNUNET_MODELS']
    
    # Check config
    if 'model_directory' in config:
        return Path(config['model_directory']).expanduser()
    
    # Default to current directory
    return Path.cwd() / "models"


def setup_nnunet_env(config: Dict):
    """Setup nnUNet environment variables"""
    paths = {
        'nnUNet_raw': config.get('nnunet_raw', '~/nnUNet_raw'),
        'nnUNet_preprocessed': config.get('nnunet_preprocessed', '~/nnUNet_preprocessed'),
        'nnUNet_results': config.get('nnunet_results', '~/nnUNet_results'),
    }
    
    for key, value in paths.items():
        expanded = Path(value).expanduser()
        expanded.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(expanded)