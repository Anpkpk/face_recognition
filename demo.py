import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt

import face_recognition as fr

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My Application")
        self.setGeometry(300, 100, 1000, 600)
        self.image_path = None

        self.initUI()

    from PyQt5.QtWidgets import QVBoxLayout

    def initUI(self):
        self.image_panel = QWidget(self)
        self.image_panel.setGeometry(0, 0, 500, 600)
        self.image_panel.setStyleSheet("background-color: lightgray;")

        image_layout = QVBoxLayout(self.image_panel)
        image_layout.setContentsMargins(10, 10, 10, 10)
        image_layout.setSpacing(10)

        self.image_label = QLabel()
        self.image_label.setFixedHeight(250)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel()
        self.label_result.setFixedHeight(30)
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.image_cropped_label = QLabel()
        self.image_cropped_label.setFixedHeight(250)
        self.image_cropped_label.setAlignment(Qt.AlignCenter)

        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.label_result)
        image_layout.addWidget(self.image_cropped_label)

        self.button_panel = QWidget(self)
        self.button_panel.setGeometry(500, 0, 500, 600)
        self.button_panel.setStyleSheet("background-color: lightgray;")

        self.select_button = QPushButton("select image", self.button_panel)
        self.select_button.setGeometry(100, 150, 200, 50)
        self.select_button.setStyleSheet("background-color: white; color: black; font-size: 16px;")
        self.select_button.clicked.connect(self.select_image)

        self.process_button = QPushButton("process image", self.button_panel)
        self.process_button.setGeometry(100, 250, 200, 50)
        self.process_button.setStyleSheet("background-color: white; color: black; font-size: 16px;")
        self.process_button.clicked.connect(self.process_image)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 
                                                   "Select Image", 
                                                   "", 
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            self.showImage(file_path)

    def showImage(self, img_path, title=""):
        img = cv2.imread(img_path)

        if img is None:
            print(f"Lỗi: không đọc được ảnh từ {img_path}")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img_rgb.dtype in [np.float32, np.float64] and img_rgb.max() > 1.0:
            img_rgb = img_rgb / 255.0

        pixmap = QPixmap(img_path)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def process_image(self):
        img_cropped = fr.crop_face(self.image_path)  
        name, dist = fr.predict_image(img_cropped)

        img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)

        height, width, channel = img_rgb.shape
        bytes_per_line = channel * width
        q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_cropped_label.setPixmap(pixmap)

        self.label_result.setText(f"{name} ({dist:.4f})")



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()
