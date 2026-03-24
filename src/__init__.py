"""
Package src: Chứa toàn bộ code chính cho hệ thống Face Recognition.
Các module chính gồm:
- config: cấu hình chung (device, path, transform...)
- engine: core logic (FaceEngine)
- gui: giao diện (PyQt5)
- dataset: custom dataset
"""

from .config import DEVICE, IMG_SIZE, MODEL_PATH, TRANSFORM
from .face_recognition import FaceEngine
from .gui import MainWindow

__all__ = [
    # Config
    "DEVICE",
    "IMG_SIZE",
    "MODEL_PATH",
    "TEMP_DIR",
    "TRANSFORM",

    # Core classes
    "FaceEngine",
    "MainWindow",

]
