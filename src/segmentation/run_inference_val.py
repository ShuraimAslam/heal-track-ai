import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.segmentation.unet import UNet

# -------------------------
# Config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ROOT = "/content/drive/MyDrive/Foot Ulcer Segmentation Challenge"
VAL_IMAGES = os.path.join(DATASET_ROOT, "validation", "images")
OUTPUT_DIR = "outputs/val_predictions"

MODEL_PATH = "checkpoints/unet_baseline.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# -------------------------
# Load model
# -------------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print(">>> Model loaded for inference")

# -------------------------
# Inference loop
# -------------------------
with torch.no_grad():
    for img_name in os.listdir(VAL_IMAGES):
        img_path = os.path.join(VAL_IMAGES, img_name)

        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        output = model(input_tensor)
        output = torch.sigmoid(output)
        mask = (output > 0.5).float()

        mask_np = mask.squeeze().cpu().numpy() * 255
        mask_img = Image.fromarray(mask_np.astype(np.uint8))

        mask_img.save(os.path.join(OUTPUT_DIR, img_name))

print(">>> Validation inference completed")