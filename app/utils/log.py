import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Use a simple StreamHandler to avoid file I/O issues during initialization
def setup(debug: bool = False) -> None:
    """
    Setup logging configuration.
    Simplified to avoid segmentation faults.
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-1s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Only use StreamHandler for now
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Reduce noise from libraries
    logging.getLogger("cv2").setLevel(logging.WARNING)
    logging.getLogger("PySide6").setLevel(logging.WARNING)
    
    # Optional: Add file logging after QApplication is created
    # This can be called later from main window if needed
    
def setup_file_logging(log_dir: str = None, debug: bool = False):
    """
    Setup file logging - call this AFTER QApplication is created.
    """
    try:
        if log_dir is None:
            # Use app data directory instead of relative path
            from PySide6.QtCore import QStandardPaths
            app_data = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
            log_dir = os.path.join(app_data, "logs")
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        log_file = os.path.join(log_dir, f"eyelog_{datetime.now().strftime('%Y%m%d')}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
        
        logging.info(f"File logging initialized: {log_file}")
        
    except Exception as e:
        logging.error(f"Failed to setup file logging: {e}")