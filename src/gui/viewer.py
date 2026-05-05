# src/gui/viewer.py
"""Custom image viewer with dark theme and consistent color mapping"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import cv2


class ImageViewer(QWidget):
    """Image viewer with dark theme and consistent color mapping"""
    
    mouse_clicked = pyqtSignal(int, int)
    image_dropped = pyqtSignal(str)
    
    # Fixed color palette for labels
    LABEL_COLORS = [
        '#000000',  # 0: black (background)
        '#FF0000',  # 1: red
        '#00FF00',  # 2: green
        '#0000FF',  # 3: blue
        '#FFFF00',  # 4: yellow
        '#00FFFF',  # 5: cyan
        '#FF00FF',  # 6: magenta
        '#FF8800',  # 7: orange
        '#8800FF',  # 8: purple
        '#00FF88',  # 9: mint
        '#FF0088',  # 10: pink
        '#88FF00',  # 11: lime
        '#0088FF',  # 12: sky blue
        '#FF8888',  # 13: light red
        '#88FF88',  # 14: light green
        '#8888FF',  # 15: light blue
    ]
    
    def __init__(self):
        super().__init__()
        
        self.image_data = None
        self.seg_data = None
        self.spacing = None
        self.axis = 2
        self.overlay_visible = True
        self.overlay_opacity = 0.5
        
        # Rotation state
        self.rotation = 0
        self.flip_h = False
        self.flip_v = False
        
        # Colormap cache
        self.cmap = None
        self.max_label = 0
        
        # Create matplotlib figure with dark theme
        self.figure = Figure(figsize=(8, 8), dpi=100, facecolor='#2b2b2b')
        self.canvas = FigureCanvas(self.figure)
        
        # Create layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        
        # Canvas
        main_layout.addWidget(self.canvas, 1)
        
        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(5, 0, 5, 5)
        button_layout.setSpacing(5)
        
        # Rotation buttons
        rot_ccw_btn = QPushButton("↺ Rotate")
        rot_ccw_btn.setMaximumWidth(80)
        rot_ccw_btn.clicked.connect(self.rotate_ccw)
        rot_ccw_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                font-size: 9px;
            }
        """)
        button_layout.addWidget(rot_ccw_btn)
        
        # Flip buttons
        flip_h_btn = QPushButton("⇄ Flip H")
        flip_h_btn.setMaximumWidth(80)
        flip_h_btn.clicked.connect(self.flip_horizontal)
        flip_h_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                font-size: 9px;
            }
        """)
        button_layout.addWidget(flip_h_btn)
        
        flip_v_btn = QPushButton("⇅ Flip V")
        flip_v_btn.setMaximumWidth(80)
        flip_v_btn.clicked.connect(self.flip_vertical)
        flip_v_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                font-size: 9px;
            }
        """)
        button_layout.addWidget(flip_v_btn)
        
        # Reset button
        reset_btn = QPushButton("Reset")
        reset_btn.setMaximumWidth(60)
        reset_btn.clicked.connect(self.reset_transforms)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                font-size: 9px;
            }
        """)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if file_path.endswith(('.nii', '.nii.gz')):
                    event.acceptProposedAction()
                    self.setStyleSheet(
                        "background-color: #444444; "
                        "border: 3px dashed #2196F3; "
                        "border-radius: 5px;"
                    )
                    return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave"""
        self.setStyleSheet("")
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop"""
        self.setStyleSheet("")
        
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if file_path.endswith(('.nii', '.nii.gz')):
                    print(f"✓ Image dropped: {file_path}")
                    self.image_dropped.emit(file_path)
                    event.acceptProposedAction()
                    return
        
        event.ignore()
    
    def create_colormap(self, max_label: int):
        """Create fixed colormap for consistent label colors"""
        num_colors = max_label + 1
        
        # Use predefined colors, add random ones if needed
        colors = self.LABEL_COLORS.copy()
        
        # Extend with random colors if needed
        while len(colors) < num_colors:
            random_color = '#' + ''.join([np.random.choice('0123456789ABCDEF') for _ in range(3)])
            colors.append(random_color)
        
        # Create colormap
        self.cmap = ListedColormap(colors[:num_colors])
        self.max_label = max_label
        
        print(f"Created colormap with {num_colors} colors")
    
    def set_image(self, image_data: np.ndarray, spacing: np.ndarray):
        """Set image data"""
        self.image_data = image_data
        self.spacing = spacing
        self.spacing[1], self.spacing[2] = self.spacing[2], self.spacing[1]

        """ resize for standard cubic resolution """
        #calculate stretch
        #stretch = np.asarray([np.max(self.spacing)/dim for dim in self.spacing])
        #new_shape = [int(stretch[i]*image_data.shape[i]) for i in range(3)]
        
        #pre allocate data for resizing
        #holder = np.zeros(np.asarray(new_shape))
        
        # find dimension to iterate over
        #iter_position = [i for i, x in enumerate(stretch) if x == 1]
        #iter_position = iter_position[0]
        
        #for i in range(new_shape[iter_position]):
        #    if iter_position == 0:
        #        holder[i,:,:] = cv2.resize(image_data[i,:,:], (new_shape[2], new_shape[1]))
        #    elif stretch[1] == 1:
        #        holder[:,i,:] = cv2.resize(image_data[:,i,:], (new_shape[2], new_shape[0]))
        #    else:
        #        holder[:,:,i] = cv2.resize(image_data[:,:,i], (new_shape[1], new_shape[0]))
 
        # Reset transforms
        self.rotation = 0
        self.flip_h = False
        self.flip_v = False
        
        #self.image_data = holder
        #print(image_data.shape)
        # Draw first slice
        self.draw_image(self.image_data[:, :, 0])
    
    def set_segmentation(self, seg_data: np.ndarray):
        """Set segmentation data and create colormap"""
        self.seg_data = seg_data
        
        # Create colormap based on max label in entire volume
        max_label = int(seg_data.max())
        self.create_colormap(max_label)
    
    def set_axis(self, axis: int):
        """Set viewing axis"""
        self.axis = axis
    
    def rotate_ccw(self):
        """Rotate view counter-clockwise by 90 degrees"""
        self.rotation = (self.rotation + 90) % 360
        print(f"Image rotated to {self.rotation}°")
    
    def flip_horizontal(self):
        """Flip view horizontally"""
        self.flip_h = not self.flip_h
        print(f"Horizontal flip: {self.flip_h}")
    
    def flip_vertical(self):
        """Flip view vertically"""
        self.flip_v = not self.flip_v
        print(f"Vertical flip: {self.flip_v}")
    
    def reset_transforms(self):
        """Reset all rotations and flips"""
        self.rotation = 0
        self.flip_h = False
        self.flip_v = False
        print("Transforms reset")
    
    def apply_transforms(self, data: np.ndarray) -> np.ndarray:
        """Apply rotation and flip transforms to 2D slice"""
        result = data.copy()
        
        # Apply flips
        if self.flip_h:
            result = np.fliplr(result)
        if self.flip_v:
            result = np.flipud(result)
        
        # Apply rotation
        if self.rotation == 90:
            result = np.rot90(result, k=1)
        elif self.rotation == 180:
            result = np.rot90(result, k=2)
        elif self.rotation == 270:
            result = np.rot90(result, k=3)
        
        return result
    
    def update_slice(self, image_slice: np.ndarray, seg_slice: np.ndarray = None):
        """Update displayed slice"""
        self.draw_image(image_slice, seg_slice)
    
    def draw_image(self, image_slice: np.ndarray, seg_slice: np.ndarray = None):
        """Draw image with optional overlay"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Apply transforms
        image_slice = self.apply_transforms(image_slice)
        
        # Set dark background
        ax.set_facecolor('#2b2b2b')
        self.figure.patch.set_facecolor('#2b2b2b')
        
        # Display image
        ax.imshow(image_slice, cmap='gray', origin='lower')
        
        # Overlay segmentation if available and visible
        if seg_slice is not None and self.overlay_visible:
            # Apply same transforms to segmentation
            seg_slice = self.apply_transforms(seg_slice)
            
            # Ensure seg_slice is integer
            seg_slice = np.asarray(seg_slice, dtype=np.int32)
            
            # Mask background (label 0)
            seg_display = np.ma.masked_where(seg_slice == 0, seg_slice)
            
            # Display with opacity using consistent colormap
            if self.cmap is not None:
                ax.imshow(
                    seg_display,
                    cmap=self.cmap,
                    alpha=self.overlay_opacity,
                    origin='lower',
                    vmin=0,
                    vmax=self.max_label
                )
        
        # Set dark text color
        ax.set_title('Image Stack (Drag & Drop Images Here)', color='white', fontsize=10)
        ax.tick_params(colors='white')
        
        # Remove axis labels for cleaner look
        ax.axis('off')
        
        self.canvas.draw()
    
    def toggle_overlay(self):
        """Toggle segmentation overlay visibility"""
        self.overlay_visible = not self.overlay_visible
        self.canvas.draw()
    
    def set_overlay_opacity(self, opacity: float):
        """Set overlay opacity"""
        self.overlay_opacity = np.clip(opacity, 0, 1)
        self.canvas.draw()
    
    def on_click(self, event):
        """Handle mouse click on canvas"""
        if event.inaxes is None:
            return
        
        x = int(event.xdata)
        y = int(event.ydata)
        
        self.mouse_clicked.emit(x, y)