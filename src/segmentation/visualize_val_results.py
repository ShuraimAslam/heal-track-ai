import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------
# Paths
# -------------------------
DATASET_ROOT = "/content/drive/MyDrive/Foot Ulcer Segmentation Challenge"

VAL_IMAGES = os.path.join(DATASET_ROOT, "validation", "images")
VAL_LABELS = os.path.join(DATASET_ROOT, "validation", "labels")
PRED_MASKS = "outputs/val_predictions"

# -------------------------
# Pick random samples
# -------------------------
image_files = sorted(os.listdir(VAL_IMAGES))
samples = random.sample(image_files, 3)  # show 3 examples

print("Visualizing samples:", samples)

# -------------------------
# Visualization
# -------------------------
for img_name in samples:
    img_path = os.path.join(VAL_IMAGES, img_name)
    gt_path = os.path.join(VAL_LABELS, img_name)
    pred_path = os.path.join(PRED_MASKS, img_name)

    image = Image.open(img_path).convert("RGB")
    gt_mask = Image.open(gt_path).convert("L")
    pred_mask = Image.open(pred_path).convert("L")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()