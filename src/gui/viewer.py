# src/gui/viewer.py
"""Custom image viewer widget using matplotlib with drag/drop support"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap


class ImageViewer(QWidget):
    """Image viewer with overlay capability and drag/drop support"""
    
    # Define signals at class level (IMPORTANT!)
    mouse_clicked = pyqtSignal(int, int)  # x, y coordinates
    image_dropped = pyqtSignal(str)  # file path
    
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
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter - show visual feedback"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if file_path.endswith(('.nii', '.nii.gz')):
                    event.acceptProposedAction()
                    # Visual feedback
                    self.setStyleSheet(
                        "background-color: #e3f2fd; "
                        "border: 3px dashed #2196F3; "
                        "border-radius: 5px;"
                    )
                    return
        
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave - reset background"""
        self.setStyleSheet("")
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop - emit signal with file path"""
        self.setStyleSheet("")  # Reset background
        
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                
                # Only accept NIFTI files
                if file_path.endswith(('.nii', '.nii.gz')):
                    print(f"✓ Image dropped: {file_path}")
                    self.image_dropped.emit(file_path)
                    event.acceptProposedAction()
                    return
        
        event.ignore()
    
    def set_image(self, image_data: np.ndarray, spacing: np.ndarray):
        """Set image data and physical spacing"""
        self.image_data = image_data
        self.spacing = spacing
        self.draw_image(image_data[:, :, 0])
    
    def set_segmentation(self, seg_data: np.ndarray):
        """Set segmentation data"""
        self.seg_data = seg_data
        self.draw_image(self.image_data[:, :, 0], self.seg_data[:, :, 0])
    
    def set_axis(self, axis: int):
        """Set viewing axis"""
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
            # Ensure seg_slice is integer
            seg_slice = np.asarray(seg_slice, dtype=np.int32)
            
            # Get unique labels
            unique_labels = np.unique(seg_slice)
            num_labels = len(unique_labels)
            
            # Generate colors for all labels
            colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
            
            # Add more colors if needed
            while len(colors) < num_labels:
                colors.append('#' + ''.join([np.random.choice('0123456789ABCDEF') for _ in range(3)]))
            
            # Create colormap
            cmap = ListedColormap(colors[:num_labels])
            
            # Mask background (label 0)
            seg_display = np.ma.masked_where(seg_slice == 0, seg_slice)
            
            # Display with opacity
            ax.imshow(
                seg_display,
                cmap=cmap,
                alpha=self.overlay_opacity,
                origin='lower',
                vmin=0,
                vmax=num_labels - 1
            )
        
        ax.set_title('Image Stack (Drag & Drop Images Here)')
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