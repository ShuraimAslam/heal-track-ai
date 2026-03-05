import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from src.segmentation.unet import UNet

# ----------------------------
# Device & Model
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/unet_baseline.pth"

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ----------------------------
# Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ----------------------------
# Post-processing (Segmentation v2)
# ----------------------------
def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    mask: binary mask (0 or 1)
    returns: refined binary mask (0 or 1)
    """

    mask = mask.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    # Fill holes inside wound
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Slightly expand boundaries (safe dilation)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ----------------------------
# Segmentation v2
# ----------------------------
def segment_wound(image: Image.Image) -> np.ndarray:
    """
    image: PIL Image
    returns: binary mask (0 or 1) after post-processing
    """

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

    # v1: raw threshold
    raw_mask = (prob_map > 0.5).astype(np.uint8)

    # v2: refined mask
    refined_mask = postprocess_mask(raw_mask)

    return refined_mask