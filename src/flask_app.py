"""
Single-file Flask app that replicates the logic from your PyQt application.
- Serves an HTML/CSS/JS frontend that captures webcam frames and shows results.
- Accepts POST requests with a camera frame (base64 PNG) for prediction.
- Accepts POST requests to register a new user (password-protected) by sending several frames.

Run:
  pip install flask torch torchvision pillow opencv-python mediapipe
  python face_recognition_flask_app.py

Open http://127.0.0.1:5000/ in your browser.

Note: This file embeds the FaceEngine and minimal config so you don't need project structure.
"""
import os
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import mediapipe as mp
from PIL import Image

import config

# --------------------------- Model classes ---------------------------
class SiameseNet(nn.Module):

    def __init__(self, embedding_dim=256):
        super(SiameseNet, self).__init__()
        mobilenet = models.mobilenet_v3_large(weights="DEFAULT")

        self.feature_extractor = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        last_channel = mobilenet.classifier[0].in_features
        self.fc = nn.Linear(last_channel, embedding_dim)
        self.dropout = nn.Dropout(0.5)

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc(x)))
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, anchor, positive, negative):
        out_anchor = self.forward_once(anchor)
        out_positive = self.forward_once(positive)
        out_negative = self.forward_once(negative)
        return out_anchor, out_positive, out_negative


class TripletLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


class FaceEngine:

    def __init__(self):
        self.reference_paths = {}
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.8
        )   
        self.save_dir = config.REGISTER_DIR
        os.makedirs(self.save_dir, exist_ok=True)

        self.set_model(config.MODEL_PATH)
        self.load_dir(config.REGISTER_DIR)

    def set_model(self, model_path):
        self.model = SiameseNet().to(config.DEVICE)
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device(config.DEVICE))
        )
        self.model.eval()


    def take_photo(self, frame, name, x=None, y=None, x2=None, y2=None):
        if isinstance(frame, Image.Image):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        if x is not None and y is not None and x2 is not None and y2 is not None:
            face_crop = frame[y:y2, x:x2]
        else:
            face_crop = frame

        save_path = os.path.join(self.save_dir, name)
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, f"{name}1.jpg")
        cv2.imwrite(filename, face_crop)

        success = cv2.imwrite(filename, face_crop)

        if success:
            print(f"[INFO] Lưu ảnh: {filename}")
            self.save_embeddings_to_txt(config.EMBEDDED_DIR)
            self.reload()

    def crop_face(self, image):
        if isinstance(image, str):  # path
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Không thể load ảnh: {image}")
        elif isinstance(image, Image.Image):  # PIL
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):  # numpy
            img = image
        else:
            raise TypeError(
                "crop_face chỉ nhận path (str), PIL.Image hoặc numpy.ndarray"
            )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.face_detector.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape

                x = max(int(bboxC.xmin * iw), 0)
                y = max(int(bboxC.ymin * ih), 0)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                x2 = min(x + w, iw)
                y2 = min(y + h, ih)
                extra_top = int(0.2 * h)
                y = max(y - extra_top, 0)

                face_crop = img[y:y2, x:x2]
                cv2.rectangle(
                    img, (x, y), (x2, y2), (0, 255, 0), 2
                )

            return Image.fromarray(
                cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            )
        
        print("No face detected in the image.")
        return None

    def load_dir(self, root_dir=config.REGISTER_DIR):
        self.reference_paths.clear()

        if any(os.path.isdir(os.path.join(root_dir, d))
               for d in os.listdir(root_dir)):

            for folder in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder)
                if os.path.isdir(folder_path):
                    embeddings = []
                    for file in os.listdir(folder_path):
                        if file.lower().endswith((".png", ".jpg", ".jpeg")):
                            img_path = os.path.join(folder_path, file)
                            img_crop = Image.open(img_path).convert("RGB")
                            img = config.TRANSFORM(img_crop).unsqueeze(0).to(config.DEVICE)

                            with torch.no_grad():
                                emb = self.model.forward_once(img).cpu()
                            embeddings.append(emb)

                    if embeddings:
                        avg_embedding = torch.mean(
                            torch.stack(embeddings), dim=0
                        )
                        self.reference_paths[folder] = avg_embedding
            self.save_embeddings_to_txt(config.EMBEDDED_DIR)
        else:
            print("Không tìm thấy thư mục con")

    def reload(self):
        self.reference_paths = self.load_embeddings_from_txt(config.EMBEDDED_DIR)

    def load_embeddings_from_txt(self, EMBEDDED_DIR):
        reference_paths = {}
        with open(EMBEDDED_DIR, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                label = parts[0]
                vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                reference_paths[label] = vector
        return reference_paths

    def save_embeddings_to_txt(self, EMBEDDED_DIR):
        with open(EMBEDDED_DIR, "w", encoding="utf-8") as f:
            for label, emb in self.reference_paths.items():
                # ép về 1D list float
                vector_str = ",".join([str(x.item()) for x in emb.view(-1)])
                f.write(f"{label},{vector_str}\n")
        print(f"Đã lưu embeddings vào {EMBEDDED_DIR}")

    def detect_and_crop(self, image_b64):
        # Decode base64 -> numpy
        try:
            img_data = base64.b64decode(image_b64.split(',')[1])
            pil_img = Image.open(BytesIO(img_data)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("Lỗi giải mã ảnh:", e)
            return None, None, None, "No face", None

        self.face_crop = None


        results = self.face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x = max(int(bboxC.xmin * iw), 0)
                y = max(int(bboxC.ymin * ih), 0)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                x2 = min(x + w, iw)
                y2 = min(y + h, ih)
                extra_top = int(0.2 * h)
                y = max(y - extra_top, 0)  
                h = max(h + extra_top, 0)

                cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

                self.face_crop = img[y:y2, x:x2]
            
                coords = (x, y, w, h)

                if self.face_crop is None or self.face_crop.size == 0:
                    continue
                break
        
        if self.face_crop is None:
            return None, None, None, "No face", None
        
        face_pil = Image.fromarray(self.face_crop)            
        best_class, best_dist = self.predict_image(face_pil)

        # Encode face crop ra base64
        _, buffer1 = cv2.imencode('.jpg', self.face_crop)
        face_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer1).decode("utf-8")

        _, buffer2 = cv2.imencode('.jpg', img)
        img = "data:image/jpeg;base64," + base64.b64encode(buffer2).decode("utf-8")


        return face_b64, img, coords, best_class, best_dist

    def convert_b64_to_pil(self, image_b64):
        img_data = base64.b64decode(image_b64)
        if "," in img_data:
            img_data = image_b64.split(",")[1]
        # Mở bằng PIL
        pil_img = Image.open(BytesIO(img_data)).convert("RGB")
        return pil_img

    def predict_image(self, img_input, threshold=0.8):
        cropped = config.TRANSFORM(img_input).unsqueeze(0).to(config.DEVICE)

        distances = []
        with torch.no_grad():
            embed_test = self.model.forward_once(cropped)

            for ref_path, ref_embedding in self.reference_paths.items():
                if ref_embedding is not None:
                    dist = torch.nn.functional.cosine_similarity(
                        embed_test, ref_embedding
                    ).item()
                    class_name = os.path.basename(ref_path)
                    distances.append((class_name, dist))
                else:
                    print(
                        "Skipping reference image due to invalid crop "
                        f"result: {ref_path}"
                    )

        if not distances:
            print("Không có reference hợp lệ sau khi crop.")
            return "unknown", None

        class_dists = {}
        for cls, dist in distances:
            class_dists.setdefault(cls, []).append(dist)

        avg_class_dists = {
            cls: sum(d) / len(d) for cls, d in class_dists.items()
        }
        sorted_dists = sorted(
            avg_class_dists.items(), key=lambda x: x[1], reverse=True
        )

        print("Khoảng cách cosin giữa ảnh test và các class:")
        for cls, d in sorted_dists:
            print(f"  {cls}: {d:.4f}")

        best_class = max(avg_class_dists, key=avg_class_dists.get)
        best_dist = avg_class_dists[best_class]

        if best_dist < threshold:
            return "Unknown", best_dist

        return best_class, best_dist

engine = FaceEngine()

# --------------------------- Flask app + template ---------------------------
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

last_predict_time = 0 

@app.route("/predict", methods=["POST"])
def predict():
    global last_predict_time
    now = time.time()

    if now - last_predict_time > 0:
        image_b64 = request.form["image"]
        face_b64, vid, coords, label, dist = engine.detect_and_crop(image_b64)
        if label == "No face":
            return jsonify(success=False, label="No face")        

        if face_b64 and coords:
            x, y, w, h = coords
            return jsonify(
                success=True,
                crop=face_b64,
                video=vid,
                x=x, y=y, width=w, height=h,
                label=label,
                distance=dist
            )
        else:
            return jsonify(success=False, message="Không tìm thấy khuôn mặt đủ lớn")
    last_predict_time = now



@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() 
    if not data or "image" not in data or "name" not in data:
        return jsonify(success=False, message="Thiếu dữ liệu"), 400
    
    name = data["name"]
    image_b64 = data["image"]

    image_reg = engine.convert_b64_to_pil(image_b64)
    face_crop = engine.crop_face(image_reg)
    engine.take_photo(face_crop, name)

    engine.load_dir()

    return jsonify(success=True, name=name)


if __name__ == '__main__':
    app.run(debug=True)
