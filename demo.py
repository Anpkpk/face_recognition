import os
import sys
import cv2
import time
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PIL import Image

from face_recognition import FaceEngine


IMG_SIZE = 160
REGISTER_DIR = r"C:\VSCode\Python\face_recognition\dataset\data"
TEMP_IMG = r"C:\VSCode\Python\face_recognition\dataset\data\temp_face.jpg"

engine = FaceEngine()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My Application")
        self.setGeometry(0, 0, 1500, 700)
        self.image_path = None
        self.current_frame = None
        self.registered_dir = None
        self.need_capture = True
        self.registered = 0
        self.last_capture_time = 0
        self.capture_delay = 3.0

        self.initUI()

        self.instructions = [
            "Nhìn thẳng",
            "Quay sang trái",
            "Quay sang phải"
        ]

        # mở camera
        self.cap = cv2.VideoCapture(0)

        # timer cập nhật frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # ===================== UI =====================
    def initUI(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f4f6f9;
            }
            QLabel {
                font-family: 'Segoe UI';
                color: #2c3e50;
            }
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #4facfe, stop:1 #00f2fe);
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 12px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #43e97b, stop:1 #38f9d7);
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #dcdfe6;
                border-radius: 10px;
                background: white;
            }
        """)
        self.set_image_labels()
        self.set_buttons()

    def set_buttons(self):
        # Panel phải
        self.button_panel = QWidget(self)
        self.button_panel.setGeometry(1000, 0, 500, 700)
        self.button_panel.setStyleSheet("""
            QWidget {
                background-color: #ecf0f1;
                border-left: 2px solid #dcdcdc;
            }
        """)

        layout = QVBoxLayout(self.button_panel)
        layout.setContentsMargins(60, 150, 60, 60)
        layout.setSpacing(40)

        # Card chứa nút
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 18px;
                border: 1px solid #dcdcdc;
                padding: 20px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 40, 20, 40)
        card_layout.setSpacing(20)

        # nút register
        self.register_button = QPushButton("Register", card)
        self.register_button.setFixedHeight(50)
        self.register_button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #4facfe, stop:1 #00f2fe);
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 12px;
                border: 1px solid #c0c0c0;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #43e97b, stop:1 #38f9d7);
            }
            QPushButton:pressed {
                background-color: #2ecc71;
            }
        """)
        self.register_button.clicked.connect(self.register)

        card_layout.addWidget(self.register_button, alignment=Qt.AlignCenter)


        layout.addWidget(card, alignment=Qt.AlignTop)
        layout.addStretch()

    def set_image_labels(self):
        # Panel trái
        self.image_panel = QWidget(self)
        self.image_panel.setGeometry(0, 0, 1000, 700)

        image_layout = QVBoxLayout(self.image_panel)
        image_layout.setContentsMargins(10, 10, 10, 10)
        image_layout.setSpacing(25)

        # camera
        self.image_label = QLabel()
        self.image_label.setFixedHeight(300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border-radius: 20px;
                border: 2px solid #e0e0e0;
            }
        """)

        # kết quả
        self.label_result = QLabel("Kết quả nhận diện")
        self.label_result.setFixedHeight(30)
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2980b9;
            }
        """)

        # label register
        self.label_register = QLabel("")
        self.label_register.setFixedHeight(30)
        self.label_register.setAlignment(Qt.AlignCenter)
        self.label_register.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2980b9;
            }
        """)
                                  

        # ảnh crop
        self.image_cropped_label = QLabel()
        self.image_cropped_label.setFixedHeight(300)
        self.image_cropped_label.setAlignment(Qt.AlignCenter)
        self.image_cropped_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border-radius: 20px;
                border: 2px solid #e0e0e0;
            }
        """)

        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.label_result)
        image_layout.addWidget(self.label_register)
        image_layout.addWidget(self.image_cropped_label)

    def process_image(self):
        img_rgb = self.current_frame
        name, dist = engine.predict_image(img_rgb)

        width, height  = img_rgb.size
        bytes_per_line = 3 * width
        q_image = QImage(
            img_rgb.tobytes(),
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)

        pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_cropped_label.setPixmap(pixmap)
        self.label_result.setText(f"{name} ({dist:.4f})")

    # ===================== REGISTER =====================
    def register(self):
        dialog = RegisterDialog()
        if dialog.exec_() == QDialog.Accepted:
            name = dialog.get_name()
            if not name:
                QMessageBox.warning(self, "Lỗi", "Tên không được để trống!")
                return

            # tạo thư mục theo tên
            self.registered_dir = os.path.join(REGISTER_DIR, name)
            os.makedirs(self.registered_dir, exist_ok=True)

            QMessageBox.information(
                self, "Thành công", f"Đã tạo thư mục: {self.registered_dir}"
            )
            self.registered = 1
        self.photo_step = 0
        engine.load_dir()

    def take_photo(self, frame, x, y, x2, y2):
        face_crop = frame[y:y2, x:x2]
        save_path = os.path.join(self.registered_dir, f"{self.photo_step+1}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

        
        self.label_register.setText(self.instructions[self.photo_step])

        if self.photo_step < 3:
            self.label_register.setText(self.instructions[self.photo_step])

            h, w, ch = face_crop.shape
            bytes_per_line = ch * w

            q_image = QImage(face_crop.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_cropped_label.setPixmap(pixmap)
        if self.photo_step >= 2:
            QMessageBox.information(
                self, "Hoàn tất", "Bạn đã đăng ký thành công"
            )
            self.label_register.setText("Đăng ký hoàn tất")
            self.registered = 0

    # ===================== UPDATE FRAME =====================
    def update_frame(self):
        with engine.mp_face_detection.FaceDetection(min_detection_confidence=0.8) as face_detection:
            ret, frame = self.cap.read()
            if not ret:
                return

            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame)

            h, w, ch = frame.shape
            bytes_per_line = ch * w

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x = max(int(bboxC.xmin * w), 0)
                    y = max(int(bboxC.ymin * h), 0)
                    self.bw = int(bboxC.width * w)
                    self.bh = int(bboxC.height * h) 
                    x2 = min(x + self.bw, w)
                    y2 = min(y + self.bh, h) 
                    
                    extra_top = int(0.2 * self.bh)
                    y = max(y - extra_top, 0)  

                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                    # nhận diện
                    if self.need_capture and (self.registered == 0):
                        if self.bw > IMG_SIZE and self.bh > IMG_SIZE:

                            # init biến cooldown
                            if not hasattr(self, "last_face"):
                                self.last_face = None
                            if not hasattr(self, "last_time"):
                                self.last_time = 0

                            now = time.time()

                            if self.last_face is None:
                                capture_new = True
                            else:
                                # so sánh vị trí mặt hiện tại và trước đó
                                lx, ly, lw, lh = self.last_face
                                dx = abs(x - lx)
                                dy = abs(y - ly)
                                dw = abs(self.bw - lw)
                                dh = abs(self.bh - lh)

                                # khác nhiều thì coi là người mới
                                different_face = (dx + dy + dw + dh) > 40
                                cooldown_ok = (now - self.last_time) > 3

                                capture_new = different_face or cooldown_ok

                            if capture_new:
                                face_crop = frame[y:y2, x:x2]
                                self.current_frame = Image.fromarray(face_crop)
                                self.image_path = TEMP_IMG
                                cv2.imwrite(self.image_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                                self.process_image()

                                # lưu trạng thái
                                self.last_face = (x, y, self.bw, self.bh)
                                self.last_time = now
        

                    # đăng ký
                    if self.registered == 1:
                        if (self.bw > 150 and self.bh > 150) and (self.photo_step <= 3):
                            current_time = time.time()
                            elapsed = current_time - self.last_capture_time

                            if elapsed >= self.capture_delay:
                                self.take_photo(frame, x, y, x2, y2)
                                self.photo_step += 1
                                self.last_capture_time = current_time
                            else:
                                remaining = int(self.capture_delay - elapsed + 1)
                                self.label_register.setText(f"{self.instructions[self.photo_step]}: {remaining} s")
                        else:
                            self.last_capture_time = time.time()  
                            self.label_register.setText("Vui lòng di chuyển lại để chụp ảnh.")
            else:
                self.last_face = None

            qimg = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio,           
                Qt.SmoothTransformation        
            )

            self.image_label.setPixmap(pixmap)
            

class RegisterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Register")
        self.setGeometry(750, 300, 350, 250)
        self.password_validated = False
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
                border-radius: 12px;
            }
            QLabel {
                font-size: 14px;
                color: #333333;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 10px;
            }
            QPushButton {
                min-width: 50px;
                padding: 6px;
                border-radius: 8px;
                border: 2px solid black; 
                background-color: white;
                color: black;
            }
            QPushButton:hover {
                background-color: #f2f2f2;
            }
        """)
        self.setUI()
        
    def setUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # --- Nhập mật khẩu ---
        pass_layout = QHBoxLayout()
        self.password_label = QLabel("Mật khẩu:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Nhập mật khẩu...")
        pass_layout.addWidget(self.password_label)
        pass_layout.addWidget(self.password_input)
        layout.addLayout(pass_layout)

        # --- Nhập tên (ẩn ban đầu) ---
        name_layout = QHBoxLayout()
        self.name_label = QLabel("Tên:")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Nhập tên...")
        name_layout.addWidget(self.name_label)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)

        self.name_input.setVisible(False)
        self.name_label.setVisible(False)

        # --- Nút OK / Cancel ---
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.validate_inputs)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def validate_inputs(self):
        if not self.password_validated:
            # kiểm tra mật khẩu lần đầu
            password = self.password_input.text()
            correct_password = "1"  # Mật khẩu mẫu
            if password == correct_password:
                self.password_validated = True
                self.password_label.setVisible(False)
                self.password_input.setVisible(False)
                self.name_label.setVisible(True)
                self.name_input.setVisible(True)
                self.name_input.setFocus()
            else:
                QMessageBox.warning(self, "Lỗi", "Mật khẩu không đúng!")
        else:
            # kiểm tra tên sau khi đã xác thực mật khẩu
            self.name = self.name_input.text()
            if not self.name:
                QMessageBox.warning(self, "Lỗi", "Tên không được để trống!")
                return
            self.accept()

    def get_name(self):
        return getattr(self, "name", "")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()
