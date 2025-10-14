import torch
from torchvision import transforms
from pathlib import Path
import os

# Thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tham số hình ảnh
IMG_SIZE = 160

# Đường dẫn model & dữ liệu
# --------------------------- Simple config ---------------------------
BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "siamese_model_tripletloss.pth"
TEMP_DIR = PROJECT_DIR / "data" / "temp_face.jpg"
REGISTER_DIR = PROJECT_DIR / "data" / "registered"
EMBEDDED_DIR = PROJECT_DIR / "output" / "embedded.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 160
TEMP_DIR = os.path.join(BASE_DIR, "temp.jpg")
# Transform ảnh
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
