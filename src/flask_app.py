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
from PIL import Image
from facenet_pytorch import MTCNN
import faiss
import pickle
import timm

from src import config

# --------------------------- Model classes  ---------------------------
# class FaceNet(nn.Module):

#     def __init__(self, embedding_dim=256):
#         super().__init__()

#         base = models.mobilenet_v3_large(
#             weights=models.MobileNet_V3_Large_Weights.DEFAULT
#         )

#         self.features = base.features
#         self.pool = nn.AdaptiveAvgPool2d(1)

#         in_feat = base.classifier[0].in_features
#         self.fc = nn.Linear(in_feat, embedding_dim)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.pool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         x = F.normalize(x, dim=1)
#         return x


class FaceNet(nn.Module):
    def __init__(self, backbone="resnet18", embedding_dim=256):
        super().__init__()

        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.features = nn.Sequential(*list(base.children())[:-1])
            in_feat = base.fc.in_features

        elif backbone == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.features = nn.Sequential(*list(base.children())[:-1])
            in_feat = base.fc.in_features

        elif backbone == "mobilenet_v2":
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            in_feat = base.last_channel

        elif backbone == "mobilenet_v3":
            base = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.DEFAULT
            )
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            in_feat = base.classifier[0].in_features

        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            in_feat = base.classifier[1].in_features

        elif backbone == "shufflenet":
            base = models.shufflenet_v2_x1_0(
                weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT
            )
            self.features = nn.Sequential(*list(base.children())[:-1])
            in_feat = base.fc.in_features

        elif backbone == "convnext_tiny":
            base = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.DEFAULT
            )
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            in_feat = base.classifier[2].in_features

        elif backbone == "mobileone":
            base = timm.create_model("mobileone_s0", pretrained=True)
            self.features = nn.Sequential(*list(base.children())[:-1])
            in_feat = base.head.fc.in_features

        elif backbone == "regnet":
            base = models.regnet_y_400mf(
                weights=models.RegNet_Y_400MF_Weights.DEFAULT
            )
            self.features = nn.Sequential(*list(base.children())[:-1])
            in_feat = base.fc.in_features

        else:
            raise ValueError("Backbone not supported")

        self.backbone = backbone
        self.fc = nn.Linear(in_feat, embedding_dim)

    def forward(self, x):

        x = self.features(x)

        if x.ndim == 4:
            x = F.adaptive_avg_pool2d(x, 1)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        x = F.normalize(x, dim=1)

        return x

# --------------------------- FaceEngine ---------------------------
class FaceEngine:

    def __init__(self):
        self.face_detector = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False,
            device=config.DEVICE
        )  
        self.save_dir = config.REGISTER_DIR
        os.makedirs(self.save_dir, exist_ok=True)

        base_path = os.path.splitext(config.EMBEDDED_DIR)[0]
        self.index_path = base_path + ".index"
        self.mapping_path = base_path + ".pkl"

        self.index = None
        self.class_names = [] # List mapping: ID (int) -> Name (str)

        self.set_model(config.MODEL_PATH)
        
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            self.reload()
        else:
            self.load_dir(config.REGISTER_DIR)

    def set_model(self, model_path):
        self.model = FaceNet('mobileone').to(config.DEVICE)
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
        filename = os.path.join(save_path, f"{name}_{int(time.time())}.jpg") # Thêm timestamp để tránh trùng

        success = cv2.imwrite(filename, face_crop)

        if success:
            print(f"[INFO] Lưu ảnh: {filename}")
            # Sau khi lưu ảnh mới, cần tính toán lại embedding và build lại index
            self.load_dir(config.REGISTER_DIR)

    def crop_face(self, image):
        # --- load image ---
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise TypeError("crop_face chỉ nhận path, PIL.Image hoặc numpy.ndarray")

        # --- detect ---
        boxes, _ = self.face_detector.detect(img)

        if boxes is None:
            print("No face detected in the image.")
            return None

        # lấy mặt lớn nhất
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        box = boxes[np.argmax(areas)]

        x1, y1, x2, y2 = map(int, box)
        w, h = img.size

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = img.crop((x1, y1, x2, y2))
        return face

    def load_dir(self, root_dir=config.REGISTER_DIR):
        """
        Quét thư mục, tính trung bình embedding cho mỗi người,
        sau đó lưu vào FAISS index.
        """
        temp_embeddings = []
        temp_names = []

        if any(os.path.isdir(os.path.join(root_dir, d)) for d in os.listdir(root_dir)):
            print("Đang quét thư mục và tạo index FAISS...")
            for folder in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder)
                if os.path.isdir(folder_path):
                    embeddings = []
                    for file in os.listdir(folder_path):
                        if file.lower().endswith((".png", ".jpg", ".jpeg")):
                            img_path = os.path.join(folder_path, file)
                            try:
                                img_crop = Image.open(img_path).convert("RGB")
                                img = config.TRANSFORM(img_crop).unsqueeze(0).to(config.DEVICE)
                                with torch.no_grad():
                                    emb = self.model.forward(img).cpu()
                                embeddings.append(emb)
                            except Exception as e:
                                print(f"Lỗi ảnh {file}: {e}")

                    if embeddings:
                        # Tính trung bình các vector của 1 người
                        avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
                        # Normalize lại vector trung bình để dùng cosine similarity
                        avg_embedding = F.normalize(avg_embedding, p=2, dim=1)
                        
                        temp_embeddings.append(avg_embedding.numpy())
                        temp_names.append(folder) # Folder name là tên người
            
            if temp_embeddings:
                # Chuyển list các tensor thành matrix numpy (N, 256)
                emb_matrix = np.vstack(temp_embeddings).astype('float32')
                self.save_faiss_index(emb_matrix, temp_names)
                self.reload()
            else:
                print("Không tìm thấy dữ liệu khuôn mặt hợp lệ.")
        else:
            print("Không tìm thấy thư mục con")

    def reload(self):
        """Load index FAISS và mapping từ đĩa lên RAM"""
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.mapping_path, 'rb') as f:
                self.class_names = pickle.load(f)
            print(f"Đã load FAISS index: {self.index.ntotal} vectors, {len(self.class_names)} classes.")
        else:
            print("Chưa có file index/mapping. Cần chạy load_dir trước.")
            self.index = None
            self.class_names = []

    def save_faiss_index(self, embedding_matrix, names):
        """Lưu matrix embedding vào FAISS và danh sách tên vào pickle"""
        dimension = embedding_matrix.shape[1] # 256
        
        # Sử dụng IndexFlatIP (Inner Product)
        # Vì vector đã được normalize, Inner Product chính là Cosine Similarity
        index = faiss.IndexFlatIP(dimension)
        index.add(embedding_matrix)
        
        # Lưu index
        faiss.write_index(index, self.index_path)
        
        # Lưu mapping tên
        with open(self.mapping_path, 'wb') as f:
            pickle.dump(names, f)
            
        print(f"Đã lưu FAISS index vào {self.index_path}")

    def detect_and_crop(self, image_b64):
        # Decode base64 -> PIL
        try:
            img_data = base64.b64decode(image_b64.split(',')[1])
            pil_img = Image.open(BytesIO(img_data)).convert("RGB")
        except Exception as e:
            print("Lỗi giải mã ảnh:", e)
            return None, None, None, "No face", None

        # ---- MTCNN detect ----
        boxes, probs = self.face_detector.detect(pil_img)

        if boxes is None:
            return None, None, None, "No face", None

        # lấy face lớn nhất
        areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
        box = boxes[np.argmax(areas)]

        x1, y1, x2, y2 = map(int, box)
        w_img, h_img = pil_img.size

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        face_pil = pil_img.crop((x1, y1, x2, y2))
        coords = (x1, y1, x2 - x1, y2 - y1)

        # ---- predict using FAISS ----
        best_class, best_dist = self.predict_image(face_pil)

        # ---- Encode face crop ----
        face_np = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)
        _, buffer1 = cv2.imencode(".jpg", face_np)
        face_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer1).decode("utf-8")

        # ---- Draw bbox lên ảnh gốc ----
        img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Vẽ tên lên ảnh
        cv2.putText(img_np, f"{best_class} ({best_dist:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buffer2 = cv2.imencode(".jpg", img_np)
        img = "data:image/jpeg;base64," + base64.b64encode(buffer2).decode("utf-8")

        return face_b64, img, coords, best_class, best_dist

    def convert_b64_to_pil(self, img_data):
        if isinstance(img_data, bytes):
            img_data = img_data.decode("utf-8")
        if "," in img_data:
            img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return img

    def predict_image(self, img_input, threshold=0.7):
        if self.index is None or self.index.ntotal == 0:
            return "Unknown", 0.0

        cropped = config.TRANSFORM(img_input).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            # Lấy vector embedding
            embed_test = self.model.forward(cropped).cpu().numpy().astype('float32')

        # ---- FAISS Search ----
        distances, indices = self.index.search(embed_test, k=5)
        
        best_dist = distances[0][0] # Similarity score
        best_index = indices[0][0]

        if best_index == -1: # Không tìm thấy
            return "Unknown", 0.0

        best_class = self.class_names[best_index]

        print(f"Predict: {best_class} - Score: {best_dist:.4f}")

        dists = distances[0]
        idxs  = indices[0]
        for rank, (dist, idx) in enumerate(zip(dists, idxs), 1):
            if idx == -1:
                continue
            name = self.class_names[idx]
            print(f"{rank}. {name} - Score: {dist:.4f}")

        if best_dist < threshold:
            return "Unknown", float(best_dist)

        return best_class, float(best_dist)

engine = FaceEngine()
engine.load_dir()

# --------------------------- Flask app (Giữ nguyên) ---------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

last_predict_time = 0 

@app.route("/predict", methods=["POST"])
def predict():
    global last_predict_time
    now = time.time()

    if now - last_predict_time > 0: # Bỏ debounce hoặc giữ tùy ý
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
    
    if face_crop is None:
         return jsonify(success=False, message="Không tìm thấy mặt để đăng ký"), 400

    engine.take_photo(face_crop, name)

    return jsonify(success=True, name=name)


if __name__ == '__main__':
    app.run(debug=True)