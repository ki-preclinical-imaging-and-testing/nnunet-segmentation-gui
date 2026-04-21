# src/gui/previous_prediction_dialog.py
"""Dialog for handling previous predictions"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from pathlib import Path


class PreviousPredictionDialog(QDialog):
    """Dialog asking user to load or overwrite previous prediction"""
    
    LOAD = 1
    OVERWRITE = 2
    CANCEL = 0
    
    def __init__(self, pred_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Previous Prediction Found")
        self.setGeometry(300, 300, 500, 200)
        self.setModal(True)
        
        self.pred_path = Path(pred_path)
        self.result = self.CANCEL
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Previous Prediction Found")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Info
        info = QLabel(
            f"A previous prediction already exists:\n\n"
            f"File: {self.pred_path.name}\n"
            f"Path: {self.pred_path.parent}\n\n"
            f"What would you like to do?"
        )
        info.setStyleSheet("color: #666; line-height: 1.6;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        layout.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        load_btn = QPushButton("Load Previous")
        load_btn.clicked.connect(self.load_previous)
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }
        """)
        button_layout.addWidget(load_btn)
        
        overwrite_btn = QPushButton("Run Prediction (Overwrite)")
        overwrite_btn.clicked.connect(self.overwrite)
        overwrite_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }
        """)
        button_layout.addWidget(overwrite_btn)
        
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
    
    def load_previous(self):
        """User chose to load previous prediction"""
        self.result = self.LOAD
        self.accept()
    
    def overwrite(self):
        """User chose to overwrite"""
        self.result = self.OVERWRITE
        self.accept()