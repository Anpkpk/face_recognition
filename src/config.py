import torch
from torchvision import transforms

# Thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tham số hình ảnh
IMG_SIZE = 160

# Đường dẫn model & dữ liệu
MODEL_PATH = r"C:\VSCode\Python\face_recognition\models\siamese_model_tripletloss.pth"
TEMP_DIR = r"C:\VSCode\Python\face_recognition\data"
REGISTER_DIR = r"C:\VSCode\Python\face_recognition\data\registered"

# Transform ảnh
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
