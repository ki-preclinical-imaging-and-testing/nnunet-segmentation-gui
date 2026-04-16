# src/gui/main_window.py
"""Main GUI window - clean and intuitive design"""
import sys
from pathlib import Path
from typing import Optional

# src/gui/main_window.py - Update imports at the top

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QComboBox,
    QFileDialog, QMessageBox, QProgressBar, QFrame,
    QDialog  # ADD THIS
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QTimer

import numpy as np
import nibabel as nib

from src.gui.viewer import ImageViewer
from src.gui.add_model_dialog import AddModelDialog  # ADD THIS
from src.core.image_handler import ImageHandler
from src.core.model_manager import ModelManager
from src.core.predictor import Predictor
from src.core.editor import BrushEditor
from src.gui.add_model_dialog import AddModelDialog

class PredictionWorker(QThread):
    """Background worker for predictions"""
    finished = pyqtSignal(str)  # Emits result path
    error = pyqtSignal(str)
    
    def __init__(self, predictor, image_path, output_dir):
        super().__init__()
        self.predictor = predictor
        self.image_path = image_path
        self.output_dir = output_dir
    
    def run(self):
        try:
            result = self.predictor.predict(self.image_path, self.output_dir)
            if result:
                self.finished.emit(result)
            else:
                self.error.emit("Prediction failed")
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, models_dir: str):
        super().__init__()
        self.setWindowTitle("nnUNet Segmentation GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Core components
        self.image_handler = ImageHandler()
        self.model_manager = ModelManager(models_dir)
        self.predictor: Optional[Predictor] = None
        self.editor = BrushEditor(brush_size=15)
        
        # State
        self.current_image_path: Optional[str] = None
        self.current_prediction_path: Optional[str] = None
        self.seg_data: Optional[np.ndarray] = None
        self.prediction_worker: Optional[PredictionWorker] = None
        self.editing = False
        self.current_tool = "paint"  # paint or erase
        
        # History for undo/redo
        self.undo_stack = []
        self.redo_stack = []
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # LEFT PANEL: Controls
        left_panel = self.create_left_panel()
        layout.addWidget(left_panel, 0)
        
        # CENTER: Image viewer
        self.viewer = ImageViewer()
        layout.addWidget(self.viewer, 1)
        
        self.show()
    
    def create_left_panel(self) -> QWidget:
        """Create left control panel"""
        panel = QWidget()
        panel.setMaximumWidth(250)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # ==================== FILE OPERATIONS ====================
        section = self.create_section("FILE")
        layout.addWidget(section)
        
        self.load_image_btn = self.create_button("Load Image", self.load_image)
        layout.addWidget(self.load_image_btn)
        
        self.image_label = QLabel("No image")
        self.image_label.setStyleSheet("color: #666; font-size: 9px;")
        layout.addWidget(self.image_label)
        
        # ==================== MODEL ====================
        section = self.create_section("MODEL")
        layout.addWidget(section)
        
        # Model selector
        layout.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        self.update_model_list()
        layout.addWidget(self.model_combo)
        
        # Load model button
        self.load_model_btn = self.create_button(
            "Load Selected Model",
            self.load_model,
            color="#4CAF50"
        )
        layout.addWidget(self.load_model_btn)
        
        # Add model button
        self.add_model_btn = self.create_button(
            "Add Model...",
            self.add_new_model,
            color="#FF9800"
        )
        layout.addWidget(self.add_model_btn)
        
        # Model status
        self.model_label = QLabel("No model loaded")
        self.model_label.setStyleSheet("color: #f44336; font-size: 9px;")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)

        # ==================== PREDICTION ====================
        section = self.create_section("PREDICTION")
        layout.addWidget(section)
        
        self.predict_btn = self.create_button("Run Prediction", self.run_prediction, color="#FF9800")
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # ==================== VIEW ====================
        section = self.create_section("VIEW")
        layout.addWidget(section)
        
        axis_layout = QHBoxLayout()
        self.axis_btn = self.create_button("Axis: Axial", self.switch_axis, color="#2196F3")
        self.axis_btn.setMaximumWidth(120)
        axis_layout.addWidget(self.axis_btn)
        layout.addLayout(axis_layout)
        
        # Slice control
        layout.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.valueChanged.connect(self.update_slice)
        layout.addWidget(self.slice_slider)
        
        self.slice_label = QLabel("0/0")
        self.slice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.slice_label)
        
        # ==================== OVERLAY ====================
        section = self.create_section("OVERLAY")
        layout.addWidget(section)
        
        self.toggle_overlay_btn = self.create_button(
            "Toggle Overlay",
            self.toggle_overlay,
            color="#9C27B0"
        )
        self.toggle_overlay_btn.setEnabled(False)
        layout.addWidget(self.toggle_overlay_btn)
        
        # Opacity
        layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.update_overlay_opacity)
        layout.addWidget(self.opacity_slider)
        
        # ==================== EDITING ====================
        section = self.create_section("EDIT")
        layout.addWidget(section)
        
        self.edit_btn = self.create_button(
            "Enable Edit",
            self.toggle_editing,
            color="#FF5722"
        )
        self.edit_btn.setEnabled(False)
        layout.addWidget(self.edit_btn)
        
        # Brush size
        layout.addWidget(QLabel("Brush Size:"))
        brush_layout = QHBoxLayout()
        
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setMinimum(5)
        self.brush_slider.setMaximum(50)
        self.brush_slider.setValue(15)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        brush_layout.addWidget(self.brush_slider)
        
        self.brush_label = QLabel("15")
        self.brush_label.setMaximumWidth(30)
        brush_layout.addWidget(self.brush_label)
        layout.addLayout(brush_layout)
        
        # Tool selection
        tool_layout = QHBoxLayout()
        self.paint_btn = self.create_button("Paint", lambda: self.set_tool("paint"), mini=True)
        self.erase_btn = self.create_button("Erase", lambda: self.set_tool("erase"), mini=True)
        tool_layout.addWidget(self.paint_btn)
        tool_layout.addWidget(self.erase_btn)
        layout.addLayout(tool_layout)
        
        # Label selection
        layout.addWidget(QLabel("Paint Label:"))
        self.label_spin = QSpinBox()
        self.label_spin.setMinimum(1)
        self.label_spin.setMaximum(20)
        self.label_spin.setValue(1)
        layout.addWidget(self.label_spin)
        
        # Undo/Redo
        undo_layout = QHBoxLayout()
        self.undo_btn = self.create_button("Undo", self.undo, mini=True)
        self.undo_btn.setEnabled(False)
        self.redo_btn = self.create_button("Redo", self.redo, mini=True)
        self.redo_btn.setEnabled(False)
        undo_layout.addWidget(self.undo_btn)
        undo_layout.addWidget(self.redo_btn)
        layout.addLayout(undo_layout)
        
        # ==================== SAVE ====================
        section = self.create_section("SAVE")
        layout.addWidget(section)
        
        self.save_btn = self.create_button("Save Segmentation", self.save_segmentation, color="#673AB7")
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)
        
        layout.addStretch()
        return panel
    
    def create_section(self, title: str) -> QLabel:
        """Create section header"""
        label = QLabel(title)
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        label.setFont(font)
        label.setStyleSheet("padding-top: 10px; border-top: 1px solid #ccc;")
        return label
    
    def create_button(
        self,
        text: str,
        callback,
        color: str = "#666",
        mini: bool = False
    ) -> QPushButton:
        """Create styled button"""
        btn = QPushButton(text)
        btn.clicked.connect(callback)
        
        if mini:
            btn.setMaximumHeight(30)
        else:
            btn.setMinimumHeight(35)
        
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
            QPushButton:pressed {{
                opacity: 0.6;
            }}
        """)
        
        return btn
    
    def update_model_list(self):
        """Update model dropdown"""
        self.model_combo.clear()
        models = self.model_manager.get_models()
        if models:
            self.model_combo.addItems(models)
        else:
            self.model_combo.addItem("No models found")
    
    def load_image(self):
        """Load NIFTI image"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load NIFTI Image",
            "",
            "NIFTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        
        if filepath:
            try:
                data, spacing, affine = self.image_handler.load_image(filepath)
                self.current_image_path = filepath
                
                # Update viewer
                self.viewer.set_image(data, spacing)
                self.slice_slider.setMaximum(data.shape[2] - 1)
                self.update_slice(0)
                
                # Update label
                name = Path(filepath).name
                shape = " × ".join(map(str, data.shape))
                self.image_label.setText(f"{name}\n{shape}")
                
                # Enable prediction if model is loaded
                if self.predictor:
                    self.predict_btn.setEnabled(True)
                
                print(f"✓ Loaded: {name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def load_model(self):
        """Load selected nnUNet model"""
        model_name = self.model_combo.currentText()
        
        if model_name == "No models found":
            QMessageBox.warning(self, "Warning", "No models available")
            return
        
        try:
            model_path = self.model_manager.get_model_path(model_name)
            
            if not model_path:
                raise ValueError(f"Model path not found for {model_name}")
            
            print(f"Loading model: {model_name}")
            print(f"Path: {model_path}")
            
            # Pass model name to predictor for output folder naming
            self.predictor = Predictor(
                str(model_path / "fold_0" / "checkpoint_best.pth"),
                model_name=model_name,
                device="cuda"
            )
            
            self.model_label.setText(f"✓ {model_name}")
            self.model_label.setStyleSheet("color: #4CAF50; font-size: 9px;")
            
            # Enable prediction if image is loaded
            if self.current_image_path:
                self.predict_btn.setEnabled(True)
            
            print(f"✓ Model loaded: {model_name}")
            
            QMessageBox.information(
                self,
                "Model Loaded",
                f"Successfully loaded: {model_name}"
            )
            
        except Exception as e:
            error_msg = str(e)
            self.model_label.setText("✗ Load failed")
            self.model_label.setStyleSheet("color: #f44336; font-size: 9px;")
            
            print(f"✗ Error: {error_msg}")
            
            QMessageBox.critical(
                self,
                "Error Loading Model",
                f"Failed to load model:\n\n{error_msg}"
            )

    def run_prediction(self):
        """Run nnUNet prediction"""
        if not self.current_image_path or not self.predictor:
            QMessageBox.warning(self, "Warning", "Load image and model first")
            return
        
        output_dir = Path(self.current_image_path).parent / "predictions"
        
        self.progress_bar.setVisible(True)
        self.predict_btn.setEnabled(False)
        
        self.prediction_worker = PredictionWorker(
            self.predictor,
            self.current_image_path,
            str(output_dir)
        )
        self.prediction_worker.finished.connect(self.on_prediction_complete)
        self.prediction_worker.error.connect(self.on_prediction_error)
        self.prediction_worker.start()

    def add_new_model(self):
        """Open dialog to add new model"""
        dialog = AddModelDialog(self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:  # Now QDialog is defined
            try:
                name, checkpoint_path, description = dialog.get_model_info()
                
                # Register model
                self.model_manager.add_model(name, checkpoint_path, description)
                
                # Refresh model list
                self.update_model_list()
                
                # Select newly added model
                index = self.model_combo.findText(name)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Model '{name}' added successfully!\n\n"
                    f"Path: {checkpoint_path}\n\n"
                    f"Description:\n{description}"
                )
                
                print(f"✓ Model added: {name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add model:\n{str(e)}")

    def on_prediction_complete(self, pred_path: str):
        """Handle successful prediction"""
        try:
            # Load prediction
            pred_img = nib.load(pred_path)
            self.seg_data = pred_img.get_fdata().astype(np.int32)
            self.current_prediction_path = pred_path
            
            # Display
            self.viewer.set_segmentation(self.seg_data)
            self.toggle_overlay_btn.setEnabled(True)
            self.edit_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            
            self.progress_bar.setVisible(False)
            self.predict_btn.setEnabled(True)
            
            # Update current slice display
            self.update_slice(self.slice_slider.value())
            
            QMessageBox.information(
                self, "Success",
                f"Prediction complete!\nSaved to: {pred_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load prediction:\n{str(e)}")
    
    def on_prediction_error(self, error: str):
        """Handle prediction error"""
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        QMessageBox.critical(self, "Prediction Error", error)
    
    def switch_axis(self):
        """Switch viewing axis"""
        axes = [0, 1, 2]
        labels = ["Sagittal", "Coronal", "Axial"]
        
        current_idx = axes.index(self.image_handler.current_axis)
        next_idx = (current_idx + 1) % 3
        
        self.image_handler.current_axis = axes[next_idx]
        self.axis_btn.setText(f"Axis: {labels[next_idx]}")
        
        # Update viewer
        if self.image_handler.image_data is not None:
            self.viewer.set_axis(axes[next_idx])
            max_slices = self.image_handler.get_shape(axes[next_idx])
            self.slice_slider.setMaximum(max_slices - 1)
            self.slice_slider.setValue(0)
            self.update_slice(0)
    
    def update_slice(self, idx: int):
        """Update displayed slice"""
        if self.image_handler.image_data is None:
            return
        
        axis = self.image_handler.current_axis
        slice_data = self.image_handler.get_slice(idx, axis)
        
        # Update segmentation slice if available
        seg_slice = None
        if self.seg_data is not None:
            if axis == 0:
                seg_slice = self.seg_data[idx, :, :].T
            elif axis == 1:
                seg_slice = self.seg_data[:, idx, :].T
            else:
                seg_slice = self.seg_data[:, :, idx]
        
        self.viewer.update_slice(slice_data, seg_slice)
        
        # Update slice label
        max_slices = self.image_handler.get_shape(axis)
        self.slice_label.setText(f"{idx + 1}/{max_slices}")
    
    def toggle_overlay(self):
        """Toggle prediction overlay visibility"""
        self.viewer.toggle_overlay()
        
        # Update button appearance
        if self.viewer.overlay_visible:
            self.toggle_overlay_btn.setText("Hide Overlay")
            self.toggle_overlay_btn.setStyleSheet("""
                QPushButton {
                    background-color: #9C27B0;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
        else:
            self.toggle_overlay_btn.setText("Show Overlay")
            self.toggle_overlay_btn.setStyleSheet("""
                QPushButton {
                    background-color: #666;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
    
    def update_overlay_opacity(self, value: int):
        """Update overlay opacity"""
        opacity = value / 100.0
        self.viewer.set_overlay_opacity(opacity)
    
    def toggle_editing(self):
        """Enable/disable editing mode"""
        self.editing = not self.editing
        
        if self.editing:
            self.edit_btn.setText("Disable Edit")
            self.edit_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
            
            # Connect mouse events
            self.viewer.mouse_clicked.connect(self.on_viewer_click)
            
            # Save undo state
            if self.seg_data is not None:
                self.undo_stack.append(self.seg_data.copy())
                self.redo_stack.clear()
                self.undo_btn.setEnabled(True)
            
            QMessageBox.information(
                self,
                "Edit Mode Enabled",
                "Click and drag on the image to paint/erase\n"
                "Use brush size slider to adjust size"
            )
        else:
            self.edit_btn.setText("Enable Edit")
            self.edit_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF5722;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
            
            # Disconnect mouse events
            try:
                self.viewer.mouse_clicked.disconnect(self.on_viewer_click)
            except:
                pass
    
    def set_tool(self, tool: str):
        """Set current editing tool"""
        self.current_tool = tool
        
        if tool == "paint":
            self.paint_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
            self.erase_btn.setStyleSheet("""
                QPushButton {
                    background-color: #999;
                    color: white;
                    border: none;
                    border-radius: 4px;
                }
            """)
        else:
            self.paint_btn.setStyleSheet("""
                QPushButton {
                    background-color: #999;
                    color: white;
                    border: none;
                    border-radius: 4px;
                }
            """)
            self.erase_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
    
    def update_brush_size(self, value: int):
        """Update brush size"""
        self.brush_label.setText(str(value))
        self.editor.set_brush_size(value)
    
    def on_viewer_click(self, x: int, y: int):
        """Handle click on viewer"""
        if not self.editing or self.seg_data is None:
            return
        
        axis = self.image_handler.current_axis
        slice_idx = self.slice_slider.value()
        
        # Convert to 3D coordinates based on axis
        if axis == 0:  # Sagittal
            z, y_3d, x_3d = slice_idx, y, x
        elif axis == 1:  # Coronal
            z, y_3d, x_3d = y, slice_idx, x
        else:  # Axial
            z, y_3d, x_3d = slice_idx, y, x
        
        # Check bounds
        if not (0 <= z < self.seg_data.shape[0] and
                0 <= y_3d < self.seg_data.shape[1] and
                0 <= x_3d < self.seg_data.shape[2]):
            return
        
        # Save to undo stack
        self.undo_stack.append(self.seg_data.copy())
        self.redo_stack.clear()
        self.undo_btn.setEnabled(True)
        self.redo_btn.setEnabled(False)
        
        # Apply edit
        if self.current_tool == "paint":
            self.seg_data = self.editor.paint_3d(
                self.seg_data,
                (z, y_3d, x_3d),
                self.label_spin.value()
            )
        else:
            self.seg_data = self.editor.erase_3d(
                self.seg_data,
                (z, y_3d, x_3d)
            )
        
        # Update display
        self.update_slice(slice_idx)
    
    def undo(self):
        """Undo last edit"""
        if len(self.undo_stack) > 0:
            self.redo_stack.append(self.seg_data.copy())
            self.seg_data = self.undo_stack.pop()
            
            self.undo_btn.setEnabled(len(self.undo_stack) > 0)
            self.redo_btn.setEnabled(True)
            
            self.update_slice(self.slice_slider.value())
    
    def redo(self):
        """Redo last undone edit"""
        if len(self.redo_stack) > 0:
            self.undo_stack.append(self.seg_data.copy())
            self.seg_data = self.redo_stack.pop()
            
            self.undo_btn.setEnabled(True)
            self.redo_btn.setEnabled(len(self.redo_stack) > 0)
            
            self.update_slice(self.slice_slider.value())
    
    def save_segmentation(self):
        """Save edited segmentation"""
        if self.seg_data is None or self.current_prediction_path is None:
            QMessageBox.warning(self, "Warning", "No segmentation to save")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Segmentation",
            str(Path(self.current_prediction_path).parent),
            "NIFTI Files (*.nii.gz);;NIFTI Files (*.nii)"
        )
        
        if filepath:
            try:
                # Load original to get affine
                pred_img = nib.load(self.current_prediction_path)
                self.image_handler.affine = pred_img.affine
                
                self.image_handler.save_image(filepath, self.seg_data)
                QMessageBox.information(self, "Success", f"Saved: {filepath}")
                print(f"✓ Saved: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")