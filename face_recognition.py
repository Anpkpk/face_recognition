import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
batch_size = 32
num_epochs = 20
lr = 0.005

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class SiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []

        for label in self.classes:
            class_path = os.path.join(root_dir, label)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append((img_path, label))

        self.label_to_images = {label: [] for label in self.classes}
        for path, label in self.image_paths:
            self.label_to_images[label].append(path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        anchor_path, anchor_label = self.image_paths[idx]
        anchor_img = Image.open(anchor_path).convert("RGB")

        positive_candidates = self.label_to_images[anchor_label].copy()
        positive_candidates.remove(anchor_path)
        if not positive_candidates:
            positive_path = anchor_path  
        else:
            positive_path = random.choice(positive_candidates)
        positive_img = Image.open(positive_path).convert("RGB")

        negative_labels = [label for label in self.classes if label != anchor_label]
        negative_label = random.choice(negative_labels)
        negative_path = random.choice(self.label_to_images[negative_label])
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.normalize(x, p=2, dim=1)
        return x


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        dist_neg = F.pairwise_distance(anchor, negative, p=2)

        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        return loss.mean()



dataset = SiameseDataset(root_dir=r"C:\VSCode\Python\face_recognition\dataset\train",
                         transform=transform)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = SiameseNet().to(device)
model.load_state_dict(torch.load(r"C:\VSCode\Python\face_recognition\siamese_model.pth"))
model.eval()


def crop_face(image):
    img = cv2.imread(image)

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(model_selection=5, min_detection_confidence=0.8) as face_detection:
        results = face_detection.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape

                x = max(int(bboxC.xmin * iw), 0)
                y = max(int(bboxC.ymin * ih) , 0)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                x2 = min(x + w, iw)
                y2 = min(y + h + 15, ih)

                face_crop = img[y:y2, x:x2]
                cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
                
    return face_crop


reference_paths = []
for root, dirs, files in os.walk(r"C:\VSCode\Python\face_recognition\dataset\train"):
    count = 0
    for file in files:
        if file.endswith(".png") or file.endswith(".jpg"):
            reference_paths.append(os.path.join(root, file))
            count += 1
            if count >= 5:
                break

def predict_image(img_input, threshold=0.8):
    if isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input).convert('RGB')
    else:
        img = Image.open(img_input).convert('RGB')

    img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        embed_anchor = model.forward(img)

    distances = []

    count = 0
    dist = 0
    for ref_path in reference_paths:
        ref_img = Image.open(ref_path).convert('RGB')
        ref_img = transform(ref_img).unsqueeze(0).to(device)

        with torch.no_grad():
            embed_ref = model.forward(ref_img)

        dist += torch.nn.functional.pairwise_distance(embed_anchor, embed_ref).item()
        count += 1
        if count % 5 == 0:
            dist = dist / 5.0
            distances.append((ref_path, dist))
            count, dist = 0, 0

    distances.sort(key=lambda x: x[1])
    name = distances[0][0].split(os.sep)[-2]
    dist = distances[0][1]

    return name, dist


