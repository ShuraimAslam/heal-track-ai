import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from src.segmentation.unet import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/unet_baseline.pth"

# Load model once
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def segment_wound(image_path: str) -> np.ndarray:
    """
    Input: path to wound image
    Output: binary mask (numpy array)
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        mask = (output > 0.5).float()

    return mask.squeeze().cpu().numpy()