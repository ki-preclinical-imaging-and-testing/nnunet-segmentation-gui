# main.py
#!/usr/bin/env python3
"""
nnUNet Segmentation GUI - Simple and Intuitive
"""
import sys
import os
from pathlib import Path

# Setup nnUNet environment
from src.utils.helpers import setup_nnunet_env, load_config, get_models_dir

config = load_config('config.yaml')
setup_nnunet_env(config)

# Import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

# Import GUI
from src.gui.main_window import MainWindow


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    try:
        # Get models directory
        models_dir = get_models_dir(config)
        
        if not Path(models_dir).exists():
            QMessageBox.warning(
                None,
                "Models Directory Not Found",
                f"Models directory not found at:\n{models_dir}\n\n"
                "Update 'model_directory' in config.yaml or set NNUNET_MODELS environment variable"
            )
        
        # Create and show main window
        window = MainWindow(str(models_dir))
        
        sys.exit(app.exec())
        
    except Exception as e:
        QMessageBox.critical(None, "Fatal Error", f"Failed to start application:\n{str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()