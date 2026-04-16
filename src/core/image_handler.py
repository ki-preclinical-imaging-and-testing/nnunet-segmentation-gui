# src/core/image_handler.py
"""Handle NIFTI image loading, saving, and manipulation"""
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class ImageHandler:
    """Load and manipulate NIFTI images with spacing"""
    
    def __init__(self):
        self.image_data = None
        self.affine = None
        self.spacing = None  # Physical spacing
        self.current_axis = 2  # Axial view
    
    def load_image(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load NIFTI image
        
        Returns:
            image_data: 3D numpy array
            spacing: physical spacing [x, y, z]
            affine: affine transformation matrix
        """
        img = nib.load(filepath)
        self.image_data = img.get_fdata()
        self.affine = img.affine
        
        # Extract spacing from affine (diagonal elements)
        self.spacing = np.abs(np.diag(self.affine)[:3])
        
        return self.image_data, self.spacing, self.affine
    
    def get_slice(self, slice_idx: int, axis: int = 2) -> np.ndarray:
        """
        Get a single slice with proper orientation
        
        Args:
            slice_idx: Slice number along axis
            axis: 0=sagittal, 1=coronal, 2=axial
        """
        if self.image_data is None:
            return None
        
        if axis == 0:  # Sagittal
            return self.image_data[slice_idx, :, :].T
        elif axis == 1:  # Coronal
            return self.image_data[:, slice_idx, :].T
        else:  # Axial
            return self.image_data[:, :, slice_idx]
    
    def set_slice(self, slice_data: np.ndarray, slice_idx: int, axis: int = 2):
        """Update a slice in the 3D volume"""
        if axis == 0:
            self.image_data[slice_idx, :, :] = slice_data.T
        elif axis == 1:
            self.image_data[:, slice_idx, :] = slice_data.T
        else:
            self.image_data[:, :, slice_idx] = slice_data
    
    def get_shape(self, axis: int = 2) -> int:
        """Get number of slices along axis"""
        if self.image_data is None:
            return 0
        return self.image_data.shape[axis]
    
    def save_image(self, output_path: str, data: Optional[np.ndarray] = None):
        """Save image to NIFTI file"""
        if data is None:
            data = self.image_data
        
        img = nib.Nifti1Image(data.astype(np.int16), self.affine)
        nib.save(img, output_path)
        print(f"✓ Saved: {output_path}")