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
    QDialog, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, QTimer
from PyQt6.QtGui import QFont

import numpy as np
import nibabel as nib

from src.gui.viewer import ImageViewer
from src.gui.add_model_dialog import AddModelDialog  # ADD THIS
from src.core.image_handler import ImageHandler
from src.core.model_manager import ModelManager
from src.core.predictor import Predictor
from src.core.editor import BrushEditor
from src.gui.add_model_dialog import AddModelDialog
from src.gui.loading_dialog import LoadingDialog
from src.gui.previous_prediction_dialog import PreviousPredictionDialog

class PredictionWorker(QObject):
    """Worker for running predictions in background thread"""
    
    finished = pyqtSignal(str)  # Emits result path
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    
    def __init__(self, predictor, image_path):
        super().__init__()
        self.predictor = predictor
        self.image_path = image_path
        self.should_stop = False
    
    def run(self):
        """Run prediction"""
        try:
            output_dir = Path(self.image_path).parent / "predictions"
            
            self.status_changed.emit("Preparing image...")
            
            # Run prediction
            self.status_changed.emit("Running model inference...")
            result = self.predictor.predict(str(self.image_path), str(output_dir))
            
            if result:
                self.status_changed.emit("Prediction complete!")
                self.finished.emit(result)
            else:
                self.error.emit("Prediction completed but output file not found")
        
        except Exception as e:
            self.error.emit(f"Prediction failed: {str(e)}")
    
    def stop(self):
        """Stop prediction"""
        self.should_stop = True


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, models_dir: str):
        super().__init__()
        self.setWindowTitle("nnUNet Segmentation GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Core components
        self.image_handler = ImageHandler()
        self.model_manager = ModelManager()
        self.predictor: Optional[Predictor] = None
        
        # State
        self.current_image_path: Optional[str] = None
        self.current_prediction_path: Optional[str] = None
        self.seg_data: Optional[np.ndarray] = None
        self.seg_confidence_data: Optional[np.ndarray] = None
        self.prediction_worker: Optional[PredictionWorker] = None
        
        # History for undo/redo (keep for potential use, but not exposed in UI)
        self.undo_stack = []
        self.redo_stack = []
        
        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout: Left controls, Right image area
        main_layout = QHBoxLayout(main_widget)
        
        # ========== LEFT PANEL: Narrow scrollable controls ==========
        left_panel = self.create_left_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(280)  # Narrower
        main_layout.addWidget(left_scroll, 0)
        
        # ========== RIGHT PANEL: Image viewer + slice navigation ==========
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        
        # Image viewer (takes most space)
        self.viewer = ImageViewer()
        self.viewer.image_dropped.connect(self.on_image_dropped)
        right_layout.addWidget(self.viewer, 1)
        
        # Slice navigation bar at bottom
        slice_nav_widget = self.create_slice_navigation()
        right_layout.addWidget(slice_nav_widget, 0)
        
        main_layout.addWidget(right_panel, 1)
        
        self.show()
    
    def create_slice_navigation(self) -> QWidget:
        """Create slice navigation bar for bottom of viewer"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Axis button
        self.axis_btn = self.create_button("Axis: Axial", self.switch_axis, color="#2196F3")
        self.axis_btn.setMaximumWidth(120)
        layout.addWidget(self.axis_btn)
        
        # Slice label
        layout.addWidget(QLabel("Slice:"))
        
        # Slice slider
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.valueChanged.connect(self.update_slice)
        layout.addWidget(self.slice_slider, 1)
        
        # Slice counter
        self.slice_label = QLabel("0/0")
        self.slice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slice_label.setMaximumWidth(50)
        layout.addWidget(self.slice_label)
        
        return widget

    def create_left_panel(self) -> QWidget:
        """Create left control panel (narrower, scrollable)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # ==================== FILE OPERATIONS ====================
        section = self.create_section("FILE")
        layout.addWidget(section)
        
        self.load_image_btn = self.create_button("Load Image", self.load_image)
        layout.addWidget(self.load_image_btn)
        
        self.image_label = QLabel("No image")
        self.image_label.setStyleSheet("color: #666; font-size: 8px;")
        self.image_label.setWordWrap(True)
        layout.addWidget(self.image_label)
        
        # ==================== MODEL ====================
        section = self.create_section("MODEL")
        layout.addWidget(section)
        
        layout.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        self.update_model_list()
        layout.addWidget(self.model_combo)
        
        self.load_model_btn = self.create_button(
            "Load Model",
            self.load_model,
            color="#4CAF50"
        )
        layout.addWidget(self.load_model_btn)
        
        self.add_model_btn = self.create_button(
            "Add Model...",
            self.add_new_model,
            color="#FF9800"
        )
        layout.addWidget(self.add_model_btn)
        
        self.model_label = QLabel("No model")
        self.model_label.setStyleSheet("color: #f44336; font-size: 8px;")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)
        
        # ==================== PREDICTION ====================
        section = self.create_section("PREDICTION")
        layout.addWidget(section)
        
        self.predict_btn = self.create_button(
            "Run Prediction",
            self.run_prediction,
            color="#FF9800"
        )
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
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
        
        layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.update_overlay_opacity)
        layout.addWidget(self.opacity_slider)
        
        # ==================== CONFIDENCE THRESHOLD (YOLO only) ====================
        self.confidence_section = QLabel("YOLO Confidence Threshold")
        confidence_section_font = QFont()
        confidence_section_font.setBold(True)
        confidence_section_font.setPointSize(10)
        self.confidence_section.setFont(confidence_section_font)
        self.confidence_section.setStyleSheet("padding-top: 10px; border-top: 1px solid #ccc;")
        self.confidence_section.setVisible(False)
        layout.addWidget(self.confidence_section)
        
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence_threshold)
        self.confidence_slider.setVisible(False)
        layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("Threshold: 0.50")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        self.confidence_label.setVisible(False)
        layout.addWidget(self.confidence_label)
        
        # ==================== SAVE ====================
        section = self.create_section("SAVE")
        layout.addWidget(section)
        
        self.save_btn = self.create_button(
            "Save Segmentation",
            self.save_segmentation,
            color="#673AB7"
        )
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
    
# src/gui/main_window.py - Update load_image and on_image_dropped

    def load_image(self):
        """Load NIFTI image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load NIFTI Image",
            "",
            "NIFTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        
        if file_path:
            try:
                # Clear previous segmentation
                self.clear_segmentation()
                
                # Reset viewer transforms
                self.viewer.reset_transforms()
                
                # Load new image
                data, spacing, affine = self.image_handler.load_image(file_path)
                self.current_image_path = file_path
                
                # Update viewer (will resample to isotropic)
                self.viewer.set_image(data, spacing)
                self.slice_slider.setMaximum(data.shape[2] - 1)
                self.update_slice(0)
                
                # Update label
                name = Path(file_path).name
                shape = " × ".join(map(str, data.shape))
                self.image_label.setText(f"{name}\n{shape}")
                
                # Enable prediction if model is loaded
                if self.predictor:
                    self.predict_btn.setEnabled(True)
                
                print(f"✓ Loaded: {name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
    
    def on_image_dropped(self, file_path: str):
        """Handle image dropped on viewer"""
        try:
            print(f"Loading dropped image: {file_path}")
            
            # Clear previous segmentation
            self.clear_segmentation()
            
            # Reset viewer transforms
            self.viewer.reset_transforms()
            
            # Load image
            data, spacing, affine = self.image_handler.load_image(file_path)
            self.current_image_path = file_path
            
            # Update viewer (will resample to isotropic)
            self.viewer.set_image(data, spacing)
            self.slice_slider.setMaximum(data.shape[2] - 1)
            self.update_slice(0)
            
            # Update label
            name = Path(file_path).name
            shape = " × ".join(map(str, data.shape))
            self.image_label.setText(f"{name}\n{shape}")
            
            # Enable prediction if model is loaded
            if self.predictor:
                self.predict_btn.setEnabled(True)
            
            print(f"✓ Loaded: {name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def clear_segmentation(self):
        """Clear current segmentation and disable related controls"""
        print("Clearing segmentation...")
        
        # Clear data
        self.seg_data = None
        self.seg_confidence_data = None
        self.current_prediction_path = None
        
        # Clear viewer
        self.viewer.seg_data = None
        self.viewer.draw_image(
            self.viewer.image_data[:, :, 0] if self.viewer.image_data is not None else np.zeros((256, 256))
        )
        
        # Disable controls
        self.toggle_overlay_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.show_confidence_controls(False)
        
        # Clear undo/redo stacks
        self.undo_stack.clear()
        self.redo_stack.clear()
        
        print("✓ Segmentation cleared")

    def load_model(self):
        """Load selected model"""
        model_name = self.model_combo.currentText()
        
        if model_name == "No models found":
            QMessageBox.warning(self, "Warning", "No models available")
            return
        
        try:
            model_data = self.model_manager.get_model(model_name)
            
            if not model_data:
                raise ValueError(f"Model data not found for {model_name}")
            
            model_type = model_data.get('type', 'nnunet')
            print(f"Loading {model_type} model: {model_name}")
            
            if model_type == 'nnunet':
                self.predictor = Predictor(model_data, model_name, "cuda")
                self.show_confidence_controls(False)  # Hide for nnUNet
            elif model_type == 'yolo':
                self.predictor = Predictor(model_data, model_name, "cuda")
                # Don't show controls yet - only show after prediction
                self.show_confidence_controls(False)
            else:
                raise NotImplementedError(f"{model_type} inference not yet implemented")
            
            self.model_label.setText(f"✓ {model_name} ({model_type})")
            self.model_label.setStyleSheet("color: #4CAF50; font-size: 9px;")
            
            if self.current_image_path:
                self.predict_btn.setEnabled(True)
            
            print(f"✓ Model loaded: {model_name}")
            
        except Exception as e:
            error_msg = str(e)
            self.model_label.setText("✗ Load failed")
            self.model_label.setStyleSheet("color: #f44336; font-size: 9px;")
            print(f"✗ Error: {error_msg}")
            QMessageBox.critical(self, "Error Loading Model", f"Failed to load model:\n\n{error_msg}")
    
    def show_confidence_controls(self, show: bool = True):
        """Show/hide confidence threshold controls"""
        self.confidence_section.setVisible(show)
        self.confidence_slider.setVisible(show)
        self.confidence_label.setVisible(show)
    
    def update_confidence_threshold(self, value: int):
        """Update segmentation based on confidence threshold"""
        # Convert slider value (0-100) to confidence (0.0-1.0)
        threshold = value / 100.0
        self.confidence_label.setText(f"Threshold: {threshold:.2f}")
        
        # If we have raw confidence data, threshold it
        if hasattr(self, 'seg_confidence_data') and self.seg_confidence_data is not None:
            # Apply threshold
            self.seg_data = (self.seg_confidence_data >= threshold).astype(np.int32)
            
            # Update viewer
            self.update_slice(self.slice_slider.value())
            
            print(f"Confidence threshold updated to {threshold:.2f}")

    def run_prediction(self):
        """Run prediction with loading dialog"""
        if not self.current_image_path or not self.predictor:
            QMessageBox.warning(self, "Warning", "Load image and model first")
            return
        
        output_dir = Path(self.current_image_path).parent / "predictions"
        prediction_model_dir = output_dir / self.predictor.model_name
        
        # Check for previous prediction
        if prediction_model_dir.exists():
            prev_pred = self.find_previous_prediction(prediction_model_dir)
            if prev_pred:
                dialog = PreviousPredictionDialog(str(prev_pred), self)
                dialog.exec()
                
                if dialog.result == PreviousPredictionDialog.LOAD:
                    # Load previous prediction
                    self.load_prediction_file(prev_pred)
                    return
                elif dialog.result == PreviousPredictionDialog.CANCEL:
                    return
                # If OVERWRITE, continue to run prediction
        
        # Show loading dialog
        self.loading_dialog = LoadingDialog(self.predictor.model_name, self)
        self.loading_dialog.cancel_requested.connect(self.cancel_prediction)
        
        # Start prediction in background thread
        from PyQt6.QtCore import QThread
        
        self.prediction_thread = QThread()
        self.prediction_worker = PredictionWorker(
            self.predictor,
            self.current_image_path
        )
        self.prediction_worker.moveToThread(self.prediction_thread)
        
        # Connect signals
        self.prediction_thread.started.connect(self.prediction_worker.run)
        self.prediction_worker.status_changed.connect(self.loading_dialog.set_status)
        self.prediction_worker.finished.connect(self.on_prediction_complete)
        self.prediction_worker.finished.connect(self.prediction_thread.quit)
        self.prediction_worker.error.connect(self.on_prediction_error)
        self.prediction_worker.error.connect(self.prediction_thread.quit)
        
        # Start thread
        self.prediction_thread.start()
        
        # Show dialog
        self.loading_dialog.exec()
    
    def find_previous_prediction(self, prediction_dir: Path):
        """Find previous prediction file in directory"""
        # Look for any .nii.gz files (excluding config files)
        nifti_files = [
            f for f in prediction_dir.glob("*.nii.gz")
            if not f.name.startswith(('dataset', 'plans', 'predict_from'))
        ]
        
        if nifti_files:
            return nifti_files[0]
        return None

# src/gui/main_window.py - Update load_prediction_file method

    def load_prediction_file(self, pred_path: Path):
        """Load existing prediction file"""
        try:
            print(f"Loading prediction: {pred_path}")
            
            # Load the segmentation
            pred_img = nib.load(str(pred_path))
            pred_data = pred_img.get_fdata()
            pred_data = np.nan_to_num(pred_data, nan=0.0)
            self.seg_data = pred_data.astype(np.int32)
            
            # Try to load confidence scores if available
            self.seg_confidence_data = None
            conf_path = pred_path.parent / pred_path.name.replace('_seg.nii.gz', '_confidence.nii.gz')
            
            if conf_path.exists():
                print(f"Loading confidence scores: {conf_path.name}")
                conf_img = nib.load(str(conf_path))
                self.seg_confidence_data = conf_img.get_fdata().astype(np.float32)
                
                # Show confidence controls if YOLO prediction
                if self.predictor and self.predictor.model_type == 'yolo':
                    self.show_confidence_controls(True)
                    self.confidence_slider.setValue(50)  # Reset to 0.5
                    print(f"Confidence range: {self.seg_confidence_data.min():.3f} - {self.seg_confidence_data.max():.3f}")
                else:
                    self.show_confidence_controls(False)
            else:
                self.show_confidence_controls(False)
            
            # Verify data
            print(f"  Shape: {self.seg_data.shape}")
            print(f"  Dtype: {self.seg_data.dtype}")
            print(f"  Min: {self.seg_data.min()}, Max: {self.seg_data.max()}")
            
            self.current_prediction_path = str(pred_path)
            
            # Display
            self.viewer.set_segmentation(self.seg_data)
            self.toggle_overlay_btn.setEnabled(True)
            self.edit_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            
            # Update current slice display
            self.update_slice(self.slice_slider.value())
            
            QMessageBox.information(
                self, "Prediction Loaded",
                f"Loaded: {pred_path.name}\n"
                f"Shape: {self.seg_data.shape}\n"
                f"Classes: {self.seg_data.min()} - {self.seg_data.max()}"
            )
            
            print(f"✓ Prediction loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading prediction: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Error",
                f"Failed to load prediction:\n{str(e)}"
            )

    def cancel_prediction(self):
        """Cancel prediction"""
        if self.prediction_thread and self.prediction_thread.isRunning():
            self.prediction_worker.stop()
            self.prediction_thread.quit()
            self.prediction_thread.wait()
            self.loading_dialog.close_dialog()
            QMessageBox.information(self, "Cancelled", "Prediction cancelled")

    def add_new_model(self):
        """Open dialog to add new model"""
        from src.gui.add_model_dialog import AddModelDialog
        
        dialog = AddModelDialog(self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                name, model_type, checkpoint_path, model_info, description = dialog.get_model_info()
                
                # Register model
                self.model_manager.add_model(
                    name,
                    model_type,
                    checkpoint_path,
                    model_info,
                    description
                )
                
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
                    f"Type: {model_type}\n"
                    f"Path: {checkpoint_path}"
                )
                
                print(f"✓ Model added: {name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add model:\n{str(e)}")

    def on_prediction_complete(self, pred_path: str):
        """Handle successful prediction"""
        try:
            # Close loading dialog
            if hasattr(self, 'loading_dialog'):
                self.loading_dialog.close_dialog()
            
            if not pred_path:
                QMessageBox.warning(self, "Error", "Prediction path is empty")
                return
            
            self.load_prediction_file(Path(pred_path))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load prediction:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def on_prediction_error(self, error: str):
        """Handle prediction error"""
        # Close loading dialog
        if hasattr(self, 'loading_dialog'):
            self.loading_dialog.close_dialog()
        
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
            self.slice_slider.setValue(max_slices//2)
            self.update_slice(0)

    def update_slice(self, idx: int):
        """Update displayed slice"""
        if self.image_handler.image_data is None:
            return
        
        axis = self.image_handler.current_axis
        
        # Get slice from RESAMPLED (isotropic) data
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