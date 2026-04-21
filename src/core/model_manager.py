# src/core/model_manager.py
"""Manage multiple model types (nnUNet, YOLO, PyTorch)"""
from pathlib import Path
from typing import List, Dict, Optional
import yaml
import re


class ModelManager:
    """Find and manage multiple model types"""
    
    SUPPORTED_TYPES = ["nnunet", "yolo", "pytorch"]
    
    CHECKPOINT_NAMES = {
        'nnunet': [
            'checkpoint_best.pth',
            'checkpoint_best.pt',
            'checkpoint_final.pth',
            'checkpoint_final.pt'
        ],
        'yolo': ['best.pt', 'last.pt'],
        'pytorch': ['model.pth', 'model.pt', 'weights.pth', 'weights.pt']
    }
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = Path(config_path)
        self.models: Dict[str, dict] = {}
        self.load_from_config()
    
    def load_from_config(self):
        """Load registered models from config file"""
        if not self.config_path.exists():
            print(f"⚠ Config file not found: {self.config_path}")
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config is None or 'registered_models' not in config:
                return
            
            for model_name, model_data in config['registered_models'].items():
                self.models[model_name] = model_data
                print(f"✓ Loaded: {model_name} ({model_data.get('type', 'unknown')})")
                
        except Exception as e:
            print(f"⚠ Error loading models from config: {e}")
    
    def get_models(self) -> List[str]:
        """Get list of available model names"""
        return sorted(self.models.keys())
    
    def get_model(self, model_name: str) -> Optional[dict]:
        """Get full model data"""
        return self.models.get(model_name)
    
    def add_model(
        self,
        model_name: str,
        model_type: str,
        checkpoint_path: str,
        model_info: dict,
        description: str = ""
    ):
        """Register a new model"""
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        checkpoint_path = Path(checkpoint_path).expanduser()
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Store model data
        self.models[model_name] = {
            'type': model_type,
            'checkpoint': str(checkpoint_path),
            'description': description,
            'model_info': model_info
        }
        
        # Save to config
        self.save_to_config()
        
        print(f"✓ Registered model: {model_name} ({model_type})")
        return model_name
    
    def save_to_config(self):
        """Save all models to config file"""
        # Load existing config
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                if config is None:
                    config = {}
            except:
                config = {}
        else:
            config = {}
        
        # Update models section
        config['registered_models'] = self.models
        
        # Save
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"✓ Saved to config")
        except Exception as e:
            print(f"✗ Error saving config: {e}")
            raise

    @staticmethod
    def extract_nnunet_info(checkpoint_path: str) -> dict:
        """Extract nnUNet parameters from checkpoint path"""
        path = Path(checkpoint_path)
        path_str = str(path)
        
        info = {
            'dataset': None,
            'config': None,
            'fold': None,
            'device': 'cuda',
            'results_dir': None
        }
        
        # Extract dataset ID
        dataset_match = re.search(r'Dataset(\d+)', path_str)
        if dataset_match:
            info['dataset'] = int(dataset_match.group(1))
        
        # Extract config
        config_match = re.search(r'nnUNetPlans__([^/\\]+)', path_str)
        if config_match:
            info['config'] = config_match.group(1)
        
        # Extract fold
        fold_match = re.search(r'fold_(\d+)', path_str)
        if fold_match:
            info['fold'] = int(fold_match.group(1))
        
        # Extract results directory
        current = Path(checkpoint_path)
        for _ in range(10):
            current = current.parent
            if current.name == "results":
                info['results_dir'] = str(current)
                break
        
        return info
    

    @staticmethod
    def extract_yolo_info(checkpoint_path: str) -> dict:
        """Extract YOLO parameters from checkpoint path"""
        path = Path(checkpoint_path)
        path_str = str(path)
        
        info = {
            'model_size': 'm',  # Default
            'device': 'cuda',
            'task': 'segment',  # detect, segment, classify, pose
        }
        
        # Try to extract model size from filename
        filename = path.name.lower()
        for size in ['n', 's', 'm', 'l', 'x']:
            if f'yolo{size}' in filename or f'-{size}.pt' in filename:
                info['model_size'] = size
                break
        
        # Check path for task hints
        if 'detect' in path_str.lower():
            info['task'] = 'detect'
        elif 'segment' in path_str.lower():
            info['task'] = 'segment'
        elif 'classify' in path_str.lower():
            info['task'] = 'classify'
        elif 'pose' in path_str.lower():
            info['task'] = 'pose'
        
        print(f"Detected YOLO settings: size={info['model_size']}, task={info['task']}")
        
        return info
    
    @staticmethod
    def extract_pytorch_info(checkpoint_path: str) -> dict:
        """Extract PyTorch model parameters"""
        info = {
            'architecture': 'unknown',
            'device': 'cuda',
            'num_classes': None
        }
        
        return info