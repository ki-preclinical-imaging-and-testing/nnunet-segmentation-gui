# src/gui/add_model_dialog.py

"""Dialog for adding new models"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt
from pathlib import Path


class AddModelDialog(QDialog):
    """Dialog to add new nnUNet model"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add nnUNet Model")
        self.setGeometry(200, 200, 500, 350)
        self.checkpoint_path = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize dialog UI"""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Register New nnUNet Model")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # Model name
        layout.addWidget(QLabel("Model Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Liver Segmentation v2")
        layout.addWidget(self.name_input)
        
        # Checkpoint path
        layout.addWidget(QLabel("Checkpoint File:"))
        layout.addWidget(QLabel("(checkpoint_best.pth or checkpoint_best.pt)"))
        path_layout = QHBoxLayout()
        
        self.path_input = QLineEdit()
        self.path_input.setReadOnly(True)
        path_layout.addWidget(self.path_input)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_checkpoint)
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 15px;
            }
        """)
        path_layout.addWidget(browse_btn)
        
        layout.addLayout(path_layout)
        
        # Description
        layout.addWidget(QLabel("Description (optional):"))
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText(
            "e.g., Segments liver from CT images\n"
            "Trained on 150+ cases\n"
            "Labels: 1=Liver, 2=Tumor, 3=Lesion"
        )
        self.desc_input.setMaximumHeight(120)
        layout.addWidget(self.desc_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        add_btn = QPushButton("Add Model")
        add_btn.clicked.connect(self.accept)
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_layout.addWidget(add_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #999;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
            }
        """)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def browse_checkpoint(self):
        """Browse for checkpoint file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Checkpoint File",
            "",
            "PyTorch Files (*.pth *.pt);;All Files (*)"
        )
        
        if filepath:
            checkpoint_path = Path(filepath)
            
            # Validate file extension
            if checkpoint_path.suffix not in ['.pth', '.pt']:
                QMessageBox.warning(
                    self,
                    "Invalid File",
                    f"File must be .pth or .pt (got {checkpoint_path.suffix})"
                )
                return
            
            # Check if it looks like a checkpoint
            if 'checkpoint' not in checkpoint_path.name.lower() and \
               'best' not in checkpoint_path.name.lower() and \
               'model' not in checkpoint_path.name.lower():
                reply = QMessageBox.question(
                    self,
                    "Confirm File",
                    f"This file doesn't look like a typical checkpoint:\n{checkpoint_path.name}\n\n"
                    "Continue anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            
            self.checkpoint_path = filepath
            self.path_input.setText(filepath)
            
            # Auto-fill name if empty
            if not self.name_input.text():
                parent_dir = checkpoint_path.parent.name
                self.name_input.setText(parent_dir)
    
    def get_model_info(self) -> tuple:
        """Get entered model information
        
        Returns:
            (model_name, checkpoint_path, description)
        """
        name = self.name_input.text().strip()
        description = self.desc_input.toPlainText().strip()
        
        if not name:
            raise ValueError("Model name is required")
        
        if not self.checkpoint_path:
            raise ValueError("Checkpoint file is required")
        
        return name, self.checkpoint_path, description