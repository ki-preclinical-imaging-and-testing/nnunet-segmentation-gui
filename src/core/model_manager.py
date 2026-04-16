# src/core/model_manager.py
"""Manage nnUNet models from local or shared directory"""
from pathlib import Path
import json
from typing import List, Dict, Optional
import yaml


class ModelManager:
    """Find and manage nnUNet models"""
    
    # Checkpoint filenames to look for
    CHECKPOINT_NAMES = [
        'checkpoint_best.pth',
        'checkpoint_best.pt',
        'checkpoint_final.pth',
        'checkpoint_final.pt',
        'model_best.pth',
        'model_best.pt'
    ]
    
    def __init__(self, models_dir: str = None, config_path: str = 'config.yaml'):
        """
        Args:
            models_dir: Path to directory containing nnUNet models (optional)
            config_path: Path to config file for storing models
        """
        self.models_dir = Path(models_dir).expanduser() if models_dir else None
        self.config_path = Path(config_path)
        self.models: Dict[str, Path] = {}
        self.scan_models()
    
    def scan_models(self):
        """Find all nnUNet model directories"""
        self.models.clear()
        
        # Load from config first
        self.load_from_config()
        
        # Then scan directory if provided
        if self.models_dir and self.models_dir.exists():
            for model_dir in self.models_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                # Look for checkpoint files
                checkpoint = self.find_checkpoint(model_dir)
                if checkpoint:
                    model_name = model_dir.name
                    if model_name not in self.models:
                        self.models[model_name] = model_dir
                        print(f"✓ Found model: {model_name} ({checkpoint.name})")
        
        if not self.models:
            print(f"⚠ No models found")
    
    @staticmethod
    def find_checkpoint(model_dir: Path) -> Optional[Path]:
        """Find checkpoint file in directory"""
        for checkpoint_name in ModelManager.CHECKPOINT_NAMES:
            checkpoint_path = model_dir / checkpoint_name
            if checkpoint_path.exists():
                return checkpoint_path
        return None
    
    def load_from_config(self):
        """Load registered models from config file"""
        if not self.config_path.exists():
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Look for models section in config
            if 'registered_models' in config:
                for model_name, model_config in config['registered_models'].items():
                    checkpoint_path = Path(model_config['checkpoint']).expanduser()
                    if checkpoint_path.exists():
                        self.models[model_name] = checkpoint_path.parent
                        print(f"✓ Loaded from config: {model_name}")
                    else:
                        print(f"⚠ Model not found: {model_name} at {checkpoint_path}")
        except Exception as e:
            print(f"⚠ Error loading models from config: {e}")
    
    def get_models(self) -> List[str]:
        """Get list of available model names"""
        return sorted(self.models.keys())
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get full path to model directory"""
        return self.models.get(model_name)

    def add_model(self, model_name: str, checkpoint_path: str, description: str = ""):
        """
        Register a new model
        
        Args:
            model_name: User-friendly name for the model
            checkpoint_path: Path to checkpoint_best.pth
            description: What this model does (optional)
        """
        checkpoint_path = Path(checkpoint_path).expanduser()
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if checkpoint_path.suffix not in ['.pth', '.pt']:
            raise ValueError(f"File must be .pth or .pt (got {checkpoint_path.suffix})")
        
        # Store checkpoint path (not just directory)
        self.models[model_name] = checkpoint_path
        
        # Save to config
        self.save_to_config(model_name, str(checkpoint_path), description)
        
        print(f"✓ Registered model: {model_name}")
        print(f"  Checkpoint: {checkpoint_path.name}")
        return model_name
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get checkpoint path to model"""
        return self.models.get(model_name)
    
    def save_to_config(self, model_name: str, checkpoint_path: str, description: str = ""):
        """Save model to config file"""
        # Load existing config
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
        
        # Create registered_models section if needed
        if 'registered_models' not in config:
            config['registered_models'] = {}
        
        # Add model
        config['registered_models'][model_name] = {
            'checkpoint': str(checkpoint_path),
            'description': description
        }
        
        # Save back to config
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Saved to config: {self.config_path}")
    
    def remove_model(self, model_name: str):
        """Remove a registered model"""
        if model_name in self.models:
            del self.models[model_name]
        
        # Remove from config
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            if 'registered_models' in config and model_name in config['registered_models']:
                del config['registered_models'][model_name]
                
                with open(self.config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Removed model: {model_name}")