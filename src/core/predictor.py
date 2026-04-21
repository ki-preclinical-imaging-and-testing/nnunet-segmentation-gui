# src/core/predictor.py
"""Inference wrapper for multiple model types (nnUNet, YOLO, PyTorch)"""
from pathlib import Path
from typing import Optional
import subprocess
import cv2
import numpy as np
import os
import sys
import shutil
import torch


class Predictor:
    """Wrapper for running inference with different model types"""
    
    def __init__(self, model_data: dict, model_name: str = None, device: str = "cuda"):
        """
        Args:
            model_data: Dictionary with type, checkpoint, model_info from config.yaml
            model_name: User-friendly name
            device: "cuda" or "cpu"
        """
        self.model_data = model_data
        self.model_name = model_name or "prediction"
        self.model_type = model_data.get('type', 'nnunet')
        self.device = device
        
        print(f"Initializing {self.model_type} predictor...")
        
        if self.model_type == 'nnunet':
            self.initialize_nnunet()
        elif self.model_type == 'yolo':
            self.initialize_yolo()
        elif self.model_type == 'pytorch':
            self.initialize_pytorch()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def initialize_nnunet(self):
        """Initialize nnUNet-specific parameters"""
        checkpoint_path = Path(self.model_data['checkpoint']).expanduser()
        model_info = self.model_data.get('model_info', {})
        
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_path.name
        self.dataset_id = str(model_info.get('dataset', '1'))
        self.config = model_info.get('config', '3d_fullres')
        self.fold = str(model_info.get('fold', 0))
        
        # Read results_dir from model_info (should be provided in yaml)
        self.results_dir = model_info.get('results_dir')
        
        if not self.results_dir:
            # Fallback: try to find it from checkpoint path
            path = checkpoint_path
            for _ in range(10):
                path = path.parent
                if path.name == "results":
                    self.results_dir = str(path)
                    break
        
        if not self.results_dir:
            raise ValueError(
                "results_dir not found in model_info and could not be auto-detected\n"
                "Please add 'results_dir' to the model_info in config.yaml"
            )
        
        print(f"✓ nnUNet initialized:")
        print(f"  Dataset: {self.dataset_id}")
        print(f"  Config: {self.config}")
        print(f"  Fold: {self.fold}")
        print(f"  Results: {self.results_dir}")
    
    def initialize_yolo(self):
        """Initialize YOLO-specific parameters"""
        checkpoint_path = Path(self.model_data['checkpoint']).expanduser()
        model_info = self.model_data.get('model_info', {})
        
        self.checkpoint_path = checkpoint_path
        self.model_size = model_info.get('model_size', 'm')
        self.task = model_info.get('task', 'detect')
        
        print(f"✓ YOLO initialized:")
        print(f"  Model: YOLOv8{self.model_size}")
        print(f"  Task: {self.task}")
    
    def initialize_pytorch(self):
        """Initialize PyTorch-specific parameters"""
        checkpoint_path = Path(self.model_data['checkpoint']).expanduser()
        model_info = self.model_data.get('model_info', {})
        
        self.checkpoint_path = checkpoint_path
        self.architecture = model_info.get('architecture', 'unknown')
        
        print(f"✓ PyTorch initialized:")
        print(f"  Architecture: {self.architecture}")
    
    def prepare_image_for_nnunet(self, image_path: str) -> str:
        """
        Prepare image for nnUNet inference
        
        nnUNet requires images to end with _0000.nii.gz for single channel
        If the file doesn't have this suffix, rename it
        
        Args:
            image_path: Path to input image
        
        Returns:
            Path to prepared image (may be renamed)
        """
        image_path = Path(image_path).expanduser().absolute()
        
        # Check if already has correct format
        if image_path.name.endswith('_0000.nii.gz'):
            return str(image_path)
        
        # Create new filename with _0000.nii.gz suffix
        stem = image_path.stem
        
        # Remove .nii if present
        if stem.endswith('.nii'):
            stem = stem[:-4]
        
        # Add _0000 suffix
        new_filename = f"{stem}_0000.nii.gz"
        new_path = image_path.parent / new_filename
        
        print(f"Renaming image for nnUNet compatibility:")
        print(f"  From: {image_path.name}")
        print(f"  To:   {new_filename}")
        
        # Copy file with new name (don't delete original)
        os.rename(image_path, new_path)
        
        return str(new_path)
    
    def predict_nnunet(self, image_path: str, output_dir: str) -> Optional[str]:
        """Run nnUNet prediction using nnUNetv2_predict command"""
        try:
            # Prepare image
            prepared_image = self.prepare_image_for_nnunet(image_path)
            
            # Create model-specific output folder
            output_path = Path(output_dir) / self.model_name
            output_path.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"nnUNet Prediction")
            print(f"{'='*60}")
            print(f"Input:  {prepared_image}")
            print(f"Output: {output_path}")
            print(f"Model:  {self.model_name}")
            
            # Set environment variables
            env = os.environ.copy()
            env['nnUNet_results'] = str(self.results_dir)
            env['nnUNet_raw'] = ""  # Not needed for inference
            env['nnUNet_preprocessed'] = ""  # Not needed for inference
            
            print(f"nnUNet_results: {env['nnUNet_results']}")
            
            # Build command: nnUNetv2_predict
            cmd = [
                "nnUNetv2_predict",
                "-i", os.path.dirname(prepared_image),
                "-o", str(output_path),
                "-d", self.dataset_id,
                "-c", self.config,
                "-f", self.fold,
                "-chk", self.checkpoint_name
            ]
            
            print(f"Command: {' '.join(cmd)}\n")
            
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
                    f"nnUNetv2_predict failed with code {result.returncode}\n"
                    f"Error: {result.stderr}"
                )
            
            # Find output file - improved detection
            pred_file = self.find_prediction_file(prepared_image, output_path)
            
            if pred_file:
                print(f"✓ Prediction saved: {pred_file.name}")
                return str(pred_file)
            
            print(f"⚠ Prediction file not found")
            print(f"Files in {output_path}:")
            for f in sorted(output_path.iterdir()):
                if f.is_file():
                    print(f"  - {f.name}")
            
            return None
            
        except subprocess.TimeoutExpired:
            print("✗ Prediction timed out (> 1 hour)")
            return None
        except FileNotFoundError:
            print("✗ nnUNetv2_predict command not found")
            print("Make sure nnunetv2 is installed: pip install nnunetv2")
            return None
        except Exception as e:
            print(f"✗ nnUNet prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def find_prediction_file(self, image_path: str, output_dir: Path) -> Optional[Path]:
        """
        Find prediction file in output directory
        
        Tries multiple naming conventions:
        1. [stem]_seg.nii.gz (standard nnUNet)
        2. [stem].nii.gz (alternate)
        3. Any other .nii.gz file (fallback)
        """
        image_path = Path(image_path)
        stem = image_path.stem
        
        # Remove _0000 suffix if present
        if stem.endswith('_0000'):
            stem = stem[:-5]
        
        # Remove .nii if present
        if stem.endswith('.nii'):
            stem = stem[:-4]
        
        print(f"Looking for prediction with stem: {stem}")
        
        # Try standard naming conventions
        for suffix in ['_seg.nii.gz', '.nii.gz']:
            pred_file = output_dir / f"{stem}{suffix}"
            print(f"  Checking: {pred_file.name}")
            if pred_file.exists():
                print(f"  ✓ Found: {pred_file.name}")
                return pred_file
        
        # Fallback: find any .nii.gz file (excluding config files)
        print(f"  Checking for any .nii.gz files...")
        nifti_files = [
            f for f in output_dir.glob("*.nii.gz")
            if not f.name.startswith(('dataset', 'plans', 'predict_from'))
        ]
        
        if nifti_files:
            # Return the first one that matches the stem
            for f in nifti_files:
                if stem in f.name:
                    print(f"  ✓ Found matching: {f.name}")
                    return f
            # If no exact match, return first non-config file
            print(f"  ✓ Found (fallback): {nifti_files[0].name}")
            return nifti_files[0]
        
        print(f"  ✗ No prediction file found")
        return None

# src/core/predictor.py - Update predict_yolo method

    def predict_yolo(self, image_path: str, output_dir: str) -> Optional[str]:
        """
        Run YOLO prediction on 3D NIFTI volume
        
        Processes each 2D slice and creates 3D segmentation output
        """
        try:
            print(f"\n{'='*60}")
            print(f"YOLO Prediction (3D)")
            print(f"{'='*60}")
            print(f"Input:  {image_path}")
            print(f"Output: {output_dir}")
            print(f"Model:  {self.model_name}")
            
            # Create output directory
            output_path = Path(output_dir) / self.model_name
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load NIFTI image
            import nibabel as nib
            from PIL import Image
            import torch
            
            img = nib.load(image_path)
            img_data = img.get_fdata()
            
            print(f"Image shape: {img_data.shape}")
            print(f"Processing {img_data.shape[0]} slices...")
            
            # Import YOLO
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError(
                    "YOLO not installed. Run: pip install ultralytics"
                )
            
            # Load model
            print(f"Loading YOLO model...")
            model = YOLO(str(self.checkpoint_path))
            
            # Initialize 3D prediction volume
            pred_volume = np.zeros_like(img_data, dtype=np.int32)
            
            # Process each slice
            for slice_idx in range(img_data.shape[0]):
                # Get slice
                slice_data = img_data[slice_idx, :, :]
                
                # Skip empty slices
                if slice_data.max() == 0:
                    continue
                
                # Normalize to 0-255 for YOLO
                if slice_data.max() > 0:
                    slice_normalized = (slice_data / slice_data.max() * 255).astype(np.uint8)
                else:
                    slice_normalized = slice_data.astype(np.uint8)
                
                # Convert grayscale to RGB using PIL (more reliable)
                slice_pil = Image.fromarray(slice_normalized, mode='L')
                slice_rgb = slice_pil.convert('RGB')
                
                # Convert PIL Image to numpy array
                slice_array = np.array(slice_rgb)
                
                print(f"  Slice {slice_idx}: shape={slice_array.shape}, dtype={slice_array.dtype}")
                
                # Run YOLO inference using PIL Image directly
                try:
                    results = model.predict(
                        source=slice_rgb,  # Pass PIL Image directly
                        verbose=False,
                        device=0 if self.device == "cuda" else "cpu",
                        conf=0.5  # Confidence threshold
                    )
                except Exception as e:
                    print(f"  Warning: Failed on slice {slice_idx}: {e}")
                    continue
                
                # Convert results to segmentation mask
                if results and len(results) > 0:
                    result = results[0]
                    
                    # Create mask from detections
                    if result.masks is not None:
                        # YOLO returns instance masks
                        masks = result.masks.data.cpu().numpy()
                        
                        # Combine all masks into single class map
                        if len(masks) > 0:
                            # Stack masks - each instance gets a class label
                            for mask_idx, mask in enumerate(masks):
                                # Resize mask to match slice
                                mask_resized = cv2.resize(
                                    mask.astype(np.float32),
                                    (slice_data.shape[1], slice_data.shape[0])
                                )
                                pred_volume[slice_idx, :, :][mask_resized > 0.2] = mask_idx + 1
                    
                    elif result.boxes is not None:
                        # Fallback: use bounding boxes if masks unavailable
                        boxes = result.boxes.xyxy.cpu().numpy()
                        for box_idx, box in enumerate(boxes):
                            x1, y1, x2, y2 = [int(b) for b in box]
                            # Bounds checking
                            x1 = max(0, min(x1, slice_data.shape[1]))
                            x2 = max(0, min(x2, slice_data.shape[1]))
                            y1 = max(0, min(y1, slice_data.shape[0]))
                            y2 = max(0, min(y2, slice_data.shape[0]))
                            if x2 > x1 and y2 > y1:
                                pred_volume[y1:y2, x1:x2, slice_idx] = box_idx + 1
                
                # Progress
                if (slice_idx + 1) % max(1, img_data.shape[2] // 10) == 0:
                    print(f"  Processed: {slice_idx + 1}/{img_data.shape[2]} slices")
            
            print(f"✓ YOLO inference complete")
            
            # Check if we got any predictions
            if pred_volume.max() == 0:
                print(f"⚠ Warning: No detections found in any slice")
            else:
                print(f"  Classes found: {np.unique(pred_volume)}")
            
            # Save as NIFTI
            pred_img = nib.Nifti1Image(pred_volume.astype(np.int16), img.affine)
            
            # Get input filename
            input_stem = Path(image_path).stem
            if input_stem.endswith('.nii'):
                input_stem = input_stem[:-4]
            
            output_file = output_path / f"{input_stem}_seg.nii.gz"
            nib.save(pred_img, output_file)
            
            print(f"✓ Prediction saved: {output_file.name}")
            return str(output_file)
            
        except Exception as e:
            print(f"✗ YOLO prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None


    def predict_pytorch(self, image_path: str, output_dir: str) -> Optional[str]:
        """Run PyTorch model prediction"""
        try:
            print(f"\n{'='*60}")
            print(f"PyTorch Prediction")
            print(f"{'='*60}")
            print(f"Input:  {image_path}")
            print(f"Output: {output_dir}")
            print(f"Model:  {self.model_name}")
            
            raise NotImplementedError("PyTorch inference not yet implemented")
            
        except Exception as e:
            print(f"✗ PyTorch prediction failed: {e}")
            return None
    
    def predict(self, image_path: str, output_dir: str) -> Optional[str]:
        """
        Run prediction based on model type
        
        Args:
            image_path: Path to input image
            output_dir: Base output directory
        
        Returns:
            Path to prediction file
        """
        if self.model_type == 'nnunet':
            return self.predict_nnunet(image_path, output_dir)
        elif self.model_type == 'yolo':
            return self.predict_yolo(image_path, output_dir)
        elif self.model_type == 'pytorch':
            return self.predict_pytorch(image_path, output_dir)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")