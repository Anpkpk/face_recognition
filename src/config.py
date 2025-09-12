import torch
from torchvision import transforms
from pathlib import Path

# Thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tham số hình ảnh
IMG_SIZE = 160

BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "siamese_model_tripletloss.pth"
TEMP_DIR = PROJECT_DIR / "data" / "temp_face.jpg"
REGISTER_DIR = PROJECT_DIR / "data" / "registered"
EMBEDDED_DIR = PROJECT_DIR / "output" / "embedded.txt"


# Transform ảnh
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
