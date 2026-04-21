# src/gui/loading_dialog.py
"""Loading dialog with progress for predictions"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from datetime import datetime


class LoadingDialog(QDialog):
    """Dialog showing prediction progress"""
    
    cancel_requested = pyqtSignal()
    
    def __init__(self, model_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Running Prediction")
        self.setGeometry(400, 300, 400, 200)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)
        
        self.model_name = model_name
        self.start_time = datetime.now()
        
        self.init_ui()
        
        # Timer to update elapsed time
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # Update every second
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Running Prediction")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Model name
        model_label = QLabel(f"Model: {self.model_name}")
        model_label.setStyleSheet("color: #666;")
        layout.addWidget(model_label)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # Progress bar (indeterminate)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # Indeterminate mode
        self.progress_bar.setMinimum(0)
        layout.addWidget(self.progress_bar)
        
        # Time info
        info_layout = QHBoxLayout()
        
        self.time_label = QLabel("Elapsed: 0s")
        self.time_label.setStyleSheet("color: #999; font-size: 10px;")
        info_layout.addWidget(self.time_label)
        
        info_layout.addStretch()
        
        self.note_label = QLabel("This may take several minutes...")
        self.note_label.setStyleSheet("color: #999; font-size: 10px;")
        info_layout.addWidget(self.note_label)
        
        layout.addLayout(info_layout)
        
        # Cancel button
        cancel_layout = QHBoxLayout()
        cancel_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMaximumWidth(100)
        self.cancel_btn.clicked.connect(self.on_cancel)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 15px;
            }
        """)
        cancel_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(cancel_layout)
        
        self.setLayout(layout)
    
    def set_status(self, status: str):
        """Update status message"""
        self.status_label.setText(status)
    
    def update_time(self):
        """Update elapsed time"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Format time
        if elapsed < 60:
            time_str = f"Elapsed: {int(elapsed)}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"Elapsed: {minutes}m {seconds}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            time_str = f"Elapsed: {hours}h {minutes}m"
        
        self.time_label.setText(time_str)
    
    def on_cancel(self):
        """Handle cancel button"""
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelling...")
        self.cancel_requested.emit()
    
    def close_dialog(self):
        """Close dialog and stop timer"""
        self.timer.stop()
        self.accept()