import os
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

from src.config import DEVICE, MODEL_PATH, REGISTER_DIR, TRANSFORM, EMBEDDED_DIR


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

        self.set_model(MODEL_PATH)
        self.load_dir(REGISTER_DIR)

    def set_model(self, model_path):
        self.model = SiameseNet().to(DEVICE)
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        self.model.eval()

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

                face_crop = img[y:y2, x:x2]
                cv2.rectangle(
                    img, (x, y), (x2, y2), (0, 255, 0), 2
                )

            return Image.fromarray(
                cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            )
        
        print("No face detected in the image.")
        return None

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
                                emb = self.model.forward_once(img).cpu()
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
            return "unknown", best_dist

        return best_class, best_dist
    



