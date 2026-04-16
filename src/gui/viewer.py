# src/gui/viewer.py
"""Custom image viewer widget using matplotlib"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, ListedColormap


class ImageViewer(QWidget):
    """Image viewer with overlay capability"""
    
    mouse_clicked = pyqtSignal(int, int)  # x, y coordinates
    
    def __init__(self):
        super().__init__()
        
        self.image_data = None
        self.seg_data = None
        self.spacing = None
        self.axis = 2
        self.overlay_visible = True
        self.overlay_opacity = 0.5
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
    
    def set_image(self, image_data: np.ndarray, spacing: np.ndarray):
        """Set image data and physical spacing"""
        self.image_data = image_data
        self.spacing = spacing
        self.draw_image(image_data[:, :, 0])  # Show first slice
    
    def set_segmentation(self, seg_data: np.ndarray):
        """Set segmentation data"""
        self.seg_data = seg_data
        self.draw_image(self.image_data[:, :, 0], self.seg_data[:, :, 0])
    
    def set_axis(self, axis: int):
        """Set viewing axis (0=sagittal, 1=coronal, 2=axial)"""
        self.axis = axis
    
    def update_slice(self, image_slice: np.ndarray, seg_slice: np.ndarray = None):
        """Update displayed slice"""
        self.draw_image(image_slice, seg_slice)
    
    def draw_image(self, image_slice: np.ndarray, seg_slice: np.ndarray = None):
        """Draw image with optional overlay"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Display image
        ax.imshow(image_slice, cmap='gray', origin='lower')
        
        # Overlay segmentation if available and visible
        if seg_slice is not None and self.overlay_visible:
            # Create colormap for segmentation
            colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
            colors.extend(['#' + ''.join([np.random.choice('0123456789ABCDEF') for _ in range(3)]) 
                          for _ in range(20)])
            
            cmap = ListedColormap(colors[:int(seg_slice.max()) + 1])
            
            # Display segmentation with transparency
            seg_display = np.ma.masked_where(seg_slice == 0, seg_slice)
            ax.imshow(seg_display, cmap=cmap, alpha=self.overlay_opacity, origin='lower')
        
        ax.set_title('Image Stack')
        ax.axis('off')
        
        self.canvas.draw()
    
    def toggle_overlay(self):
        """Toggle segmentation overlay visibility"""
        self.overlay_visible = not self.overlay_visible
        self.canvas.draw()
    
    def set_overlay_opacity(self, opacity: float):
        """Set overlay opacity (0-1)"""
        self.overlay_opacity = np.clip(opacity, 0, 1)
        self.canvas.draw()
    
    def on_click(self, event):
        """Handle mouse click on canvas"""
        if event.inaxes is None:
            return
        
        x = int(event.xdata)
        y = int(event.ydata)
        
        self.mouse_clicked.emit(x, y)