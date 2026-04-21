# src/gui/add_model_dialog.py
"""Enhanced dialog for adding models with type selection"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QTextEdit, QMessageBox,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from pathlib import Path
from src.core.model_manager import ModelManager


class AddModelDialog(QDialog):
    """Enhanced dialog to add new models"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Model")
        self.setGeometry(200, 200, 700, 600)
        self.checkpoint_path = None
        self.extracted_info = {}
        self.init_ui()
    
    def init_ui(self):
        """Initialize dialog UI"""
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        # Title
        title = QLabel("Register New Model")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # Model type selection
        layout.addWidget(QLabel("Model Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["nnunet", "yolo", "pytorch"])
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        layout.addWidget(self.type_combo)
        
        # Model name
        layout.addWidget(QLabel("Model Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Lung Tumor Segmentation")
        layout.addWidget(self.name_input)
        
        # Checkpoint path
        layout.addWidget(QLabel("Checkpoint File:"))
        path_layout = QHBoxLayout()
        
        self.path_input = QLineEdit()
        self.path_input.setReadOnly(True)
        path_layout.addWidget(self.path_input)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_checkpoint)
        browse_btn.setMaximumWidth(100)
        path_layout.addWidget(browse_btn)
        
        layout.addLayout(path_layout)
        
        # Description
        layout.addWidget(QLabel("Description:"))
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText(
            "e.g., Segments lung tumors from CT images\n"
            "Trained on 150+ cases"
        )
        self.desc_input.setMaximumHeight(80)
        layout.addWidget(self.desc_input)
        
        # Model info table (extracted parameters)
        layout.addWidget(QLabel("Model Parameters (Edit as needed):"))
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.info_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.info_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.info_table.setMaximumHeight(150)
        layout.addWidget(self.info_table)
        
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
        layout.addStretch()
        
        self.setLayout(layout)
    
    def on_type_changed(self, model_type: str):
        """Update checkpoint filter based on model type"""
        print(f"Model type changed to: {model_type}")
    
    def browse_checkpoint(self):
        """Browse for checkpoint file"""
        model_type = self.type_combo.currentText()
        
        # Set filter based on type
        if model_type == "nnunet":
            file_filter = "nnUNet Checkpoint (*.pth *.pt);;All Files (*)"
        elif model_type == "yolo":
            file_filter = "YOLO Model (*.pt);;All Files (*)"
        else:
            file_filter = "PyTorch Model (*.pth *.pt);;All Files (*)"
        
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {model_type.upper()} Checkpoint",
            "",
            file_filter
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
            
            self.checkpoint_path = filepath
            self.path_input.setText(filepath)
            
            # Auto-extract parameters
            self.extract_parameters(model_type, filepath)
            
            # Auto-fill name if empty
            if not self.name_input.text():
                parent_dir = checkpoint_path.parent.parent.name if model_type == "nnunet" else checkpoint_path.parent.name
                self.name_input.setText(parent_dir)
    
    def extract_parameters(self, model_type: str, checkpoint_path: str):
        """Extract and display model parameters"""
        self.extracted_info = {}
        
        try:
            if model_type == "nnunet":
                self.extracted_info = ModelManager.extract_nnunet_info(checkpoint_path)
            elif model_type == "yolo":
                self.extracted_info = ModelManager.extract_yolo_info(checkpoint_path)
            else:  # pytorch
                self.extracted_info = ModelManager.extract_pytorch_info(checkpoint_path)
            
            # Display in table
            self.update_info_table()
            
            print(f"✓ Extracted {model_type} parameters: {self.extracted_info}")
            
        except Exception as e:
            print(f"⚠ Error extracting parameters: {e}")
    
    def update_info_table(self):
        """Update the parameters table"""
        self.info_table.setRowCount(0)
        
        for key, value in self.extracted_info.items():
            row = self.info_table.rowCount()
            self.info_table.insertRow(row)
            
            # Parameter name
            name_item = QTableWidgetItem(key)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.info_table.setItem(row, 0, name_item)
            
            # Parameter value (editable)
            value_item = QTableWidgetItem(str(value))
            self.info_table.setItem(row, 1, value_item)
    
    def get_model_info(self) -> tuple:
        """Get entered model information
        
        Returns:
            (model_name, model_type, checkpoint_path, model_info, description)
        """
        name = self.name_input.text().strip()
        model_type = self.type_combo.currentText()
        description = self.desc_input.toPlainText().strip()
        
        if not name:
            raise ValueError("Model name is required")
        
        if not self.checkpoint_path:
            raise ValueError("Checkpoint file is required")
        
        # Get edited parameters from table
        model_info = {}
        for row in range(self.info_table.rowCount()):
            key = self.info_table.item(row, 0).text()
            value = self.info_table.item(row, 1).text()
            
            # Try to convert to appropriate type
            try:
                # Try int first
                model_info[key] = int(value)
            except ValueError:
                # Try float
                try:
                    model_info[key] = float(value)
                except ValueError:
                    # Keep as string
                    model_info[key] = value
        
        return name, model_type, self.checkpoint_path, model_info, description