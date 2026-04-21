# src/core/editor.py
"""3D brush and eraser for editing segmentations"""
import numpy as np
from scipy.ndimage import distance_transform_edt


class BrushEditor:
    """3D brush-based editor for segmentations"""
    
    def __init__(self, brush_size: int = 15):
        self.brush_size = brush_size
        self.brush_mask_3d = self._create_brush_mask()
    
    def _create_brush_mask(self) -> np.ndarray:
        """Create 3D spherical brush mask"""
        s = self.brush_size
        z, y, x = np.ogrid[-s:s+1, -s:s+1, -s:s+1]
        mask = x*x + y*y + z*z <= s*s
        return mask.astype(np.uint8)
    
    def set_brush_size(self, size: int):
        """Update brush size"""
        self.brush_size = size
        self.brush_mask_3d = self._create_brush_mask()

    def paint_3d(
        self,
        seg_volume: np.ndarray,
        center: tuple,
        label: int
    ) -> np.ndarray:
        """
        Paint label at 3D position
        
        Args:
            seg_volume: 3D segmentation array (integer class labels)
            center: (z, y, x) center position
            label: Label value to paint (integer)
        """
        # Ensure proper types
        result = np.asarray(seg_volume, dtype=np.int32).copy()
        label = int(label)
        
        z, y, x = [int(c) for c in center]
        s = self.brush_size
        
        # Bounds
        z_min, z_max = max(0, z-s), min(result.shape[0], z+s+1)
        y_min, y_max = max(0, y-s), min(result.shape[1], y+s+1)
        x_min, x_max = max(0, x-s), min(result.shape[2], x+s+1)
        
        # Adjust mask for boundaries
        mask_z_start = z_min - (z - s)
        mask_z_end = mask_z_start + (z_max - z_min)
        mask_y_start = y_min - (y - s)
        mask_y_end = mask_y_start + (y_max - y_min)
        mask_x_start = x_min - (x - s)
        mask_x_end = mask_x_start + (x_max - x_min)
        
        mask = self.brush_mask_3d[
            mask_z_start:mask_z_end,
            mask_y_start:mask_y_end,
            mask_x_start:mask_x_end
        ]
        
        result[z_min:z_max, y_min:y_max, x_min:x_max][mask == 1] = label
        
        return result.astype(np.int32)
        
    def erase_3d(
        self,
        seg_volume: np.ndarray,
        center: tuple
    ) -> np.ndarray:
        """Erase (set to 0) at 3D position"""
        return self.paint_3d(seg_volume, center, label=0)