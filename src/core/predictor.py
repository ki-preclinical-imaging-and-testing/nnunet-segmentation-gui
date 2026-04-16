# src/core/predictor.py
"""nnUNetv2 inference wrapper"""
from pathlib import Path
from typing import Optional, Tuple
import subprocess
import os
import re
import sys
import glob


class Predictor:
    """Wrapper for nnUNetv2_predict"""
    
    def __init__(self, checkpoint_path: str, model_name: str = None, device: str = "cuda"):
        """
        Args:
            checkpoint_path: Full path to checkpoint_best.pth
            model_name: User-friendly name for output folder
            device: "cuda" or "cpu"
        """
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.model_name = model_name or "prediction"
        self.device = device
        
        # Extract parameters from checkpoint path
        self.dataset_id, self.config, self.fold = self._extract_params()
        
        # Get results directory
        self.results_dir = self._get_results_dir()
        
        print(self.checkpoint_path)
        print(self.model_name)
        print(self.device)
        print(self.results_dir)

        self.initialize()
    
    def _extract_params(self) -> Tuple[str, str, str]:
        """Extract dataset ID, config, and fold from checkpoint path"""
        path_parts = self.checkpoint_path.parts
        
        dataset_id = None
        config = None
        fold = None
        
        for part in path_parts:
            if part.startswith("Dataset"):
                match = re.search(r'Dataset(\d+)', part)
                if match:
                    dataset_id = match.group(1).lstrip('0') or '0'
            
            if "nnUNetPlans__" in part:
                config = part.split("nnUNetPlans__")[-1]
            
            if part.startswith("fold_"):
                fold = part.split("_")[-1]
        
        if not all([dataset_id, config, fold]):
            raise ValueError(
                f"Could not extract parameters from: {self.checkpoint_path}\n"
                f"Dataset: {dataset_id}, Config: {config}, Fold: {fold}"
            )
        
        print(f"✓ Extracted parameters:")
        print(f"  Dataset: {dataset_id}")
        print(f"  Config: {config}")
        print(f"  Fold: {fold}")
        
        return dataset_id, config, fold
    
    def _get_results_dir(self) -> Path:
        """Find the results directory (parent of DatasetXXX folders)"""
        current = self.checkpoint_path.parent
        
        # Navigate up until we find a directory containing Dataset folders
        for _ in range(15):
            current = current.parent
            
            try:
                # Check if this directory contains Dataset* folders
                has_dataset = any(
                    d.is_dir() and d.name.startswith('Dataset') 
                    for d in current.iterdir()
                )
                if has_dataset:
                    print(f"✓ Found results directory: {current}")
                    return current
            except PermissionError:
                continue
        
        raise ValueError(
            f"Could not find results directory containing Dataset folders\n"
            f"Starting from: {self.checkpoint_path}"
        )
    
    def initialize(self):
        """Initialize predictor"""
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Results dir: {self.results_dir}")
        print(f"Model name: {self.model_name}")
        print("✓ Ready for prediction")
    
    def predict(self, image_path: str, output_base_dir: str) -> Optional[str]:
        """Run prediction"""
        try:
            # Normalize paths
            image_path = str(Path(image_path).expanduser().absolute())
            output_base_dir = str(Path(output_base_dir).expanduser().absolute())
            
            # Create model-specific output folder
            output_dir = str(Path(output_base_dir) / self.model_name)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"nnUNetv2 Prediction")
            print(f"{'='*60}")
            print(f"Input:  {image_path}")
            print(f"Output: {output_dir}")
            print(f"Dataset: {self.dataset_id}")
            print(f"Config: {self.config}")
            print(f"Fold: {self.fold}")
            print(f"Results: {self.results_dir}")
            
            # Verify input exists
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Set environment
            env = os.environ.copy()
            env['nnUNet_results'] = str(self.results_dir)
            
            # Build command
            cmd = [
                "nnUNetv2_predict",
                "-i", os.path.dirname(image_path),
                "-o", output_dir,
                "-d", self.dataset_id,
                "-c", self.config,
                "-f", self.fold,
                "-chk", "checkpoint_best.pth"
            ]
            
            print(f"\nCommand: {' '.join(cmd)}\n")
            
            # Run prediction
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            # Print output
            if result.stdout:
                print("Output:")
                print(result.stdout)
            
            if result.stderr:
                print("Stderr:")
                print(result.stderr)
            
            if result.returncode != 0:
                raise RuntimeError(
                    f"Prediction failed with code {result.returncode}\n"
                    f"Error: {result.stderr}"
                )
            
            # Find output file

            pred_file = glob.glob(os.path.join(output_dir,'*.nii.gz'))[0]
            print(pred_file)
            return pred_file 

            #stem = Path(image_path).stem
            #if stem.endswith('.nii'):
            #    stem = stem[:-8]
            #    print(f'stem : {stem}')
            
            #output_path = Path(output_dir)
            #print(f' output_path : {output_path}')
            # Search for prediction file
            #for suffix in ['.nii.gz', '.nii']:
            #    pred_file = output_path / f"{stem}{suffix}"
            #    print(f' looking for : {pred_file}')
            #    if pred_file.exists():
            #        print(f"✓ Prediction saved: {pred_file.name}")
            #        return str(pred_file)
            
            # Fallback
            #seg_files = list(output_path.glob("*_seg.nii.gz")) + list(output_path.glob("*_seg.nii"))
            #if seg_files:
            #    pred_file = seg_files[0]
            #    print(f"✓ Found: {pred_file.name}")
            #    return str(pred_file)
            
            #print(f"⚠ Prediction file not found")
            #return None
                
        except subprocess.TimeoutExpired:
            print("✗ Prediction timed out")
            return None
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None