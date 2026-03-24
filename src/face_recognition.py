import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from facenet_pytorch import MTCNN

from src.config import DEVICE, MODEL_PATH, REGISTER_DIR, TRANSFORM, EMBEDDED_DIR


class FaceNet(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()

        base = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        in_feat = base.classifier[0].in_features
        self.fc = nn.Linear(in_feat, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = F.normalize(x, dim=1)
        return x


class ArcFaceLoss(nn.Module):

    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.5):
        super().__init__()

        self.s = s
        self.m = m

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_dim)
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)

        cosine = F.linear(embeddings, weight)

        theta = torch.acos(torch.clamp(cosine, -1+1e-7, 1-1e-7))
        target_logits = torch.cos(theta + self.m)

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()

        logits = cosine * (1 - one_hot) + target_logits * one_hot
        logits *= self.s

        loss = F.cross_entropy(logits, labels)
        return loss


class FaceEngine:

    def __init__(self):
        self.reference_paths = {}
        self.face_detector = MTCNN(
            image_size=160,
            margin=20,          # padding quanh mặt
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False,
            device=DEVICE
        )  

        self.set_model(MODEL_PATH)
        self.load_dir(REGISTER_DIR)

    def set_model(self, model_path):
        self.model = FaceNet().to(DEVICE)
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device(DEVICE))
        )
        self.model.eval()

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
        boxes, _ = self.detector.detect(img)

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

    def load_dir(self, root_dir=REGISTER_DIR):
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
                            img = TRANSFORM(img_crop).unsqueeze(0).to(DEVICE)

                            with torch.no_grad():
                                emb = self.model.forward(img).cpu()
                            embeddings.append(emb)

                    if embeddings:
                        avg_embedding = torch.mean(
                            torch.stack(embeddings), dim=0
                        )
                        self.reference_paths[folder] = avg_embedding
            self.save_embeddings_to_txt(EMBEDDED_DIR)
        else:
            print("Không tìm thấy thư mục con")

    def reload(self):
        self.reference_paths = self.load_embeddings_from_txt(EMBEDDED_DIR)



    def load_embeddings_from_txt(self, txt_path):
        reference_paths = {}
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                label = parts[0]
                vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                reference_paths[label] = vector
        return reference_paths

    def save_embeddings_to_txt(self, txt_path):
        with open(txt_path, "w", encoding="utf-8") as f:
            for label, emb in self.reference_paths.items():
                # ép về 1D list float
                vector_str = ",".join([str(x.item()) for x in emb.view(-1)])
                f.write(f"{label},{vector_str}\n")
        print(f"Đã lưu embeddings vào {txt_path}")



    def predict_image(self, img_input, threshold=0.8):
        cropped = TRANSFORM(img_input).unsqueeze(0).to(DEVICE)

        distances = []
        with torch.no_grad():
            embed_test = self.model.forward(cropped)

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
            return "unknown", best_dist

        return best_class, best_dist