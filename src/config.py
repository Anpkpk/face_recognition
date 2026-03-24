import torch
from torchvision import transforms
from pathlib import Path
import os

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Image size
IMG_SIZE = 160

# Path
BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "SiameseNet_ArcFace" / "best_model_mobile_one.pth"
TEMP_DIR = PROJECT_DIR / "data" / "temp_face.jpg"
REGISTER_DIR = PROJECT_DIR / "data" / "registered"
EMBEDDED_DIR = PROJECT_DIR / "output" / "embedded.txt"
TEMP_DIR = os.path.join(BASE_DIR, "temp.jpg")

TRANSFORM = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])