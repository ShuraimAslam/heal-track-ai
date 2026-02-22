print(">>> train_unet.py STARTED")

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.dataset import WoundSegmentationDataset
from src.segmentation.unet import UNet

print(">>> Imports completed")

# -------------------------
# Config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ROOT = "D:/AIML/dataset/Foot Ulcer Segmentation Challenge"

TRAIN_IMAGES = os.path.join(DATASET_ROOT, "train", "images")
TRAIN_LABELS = os.path.join(DATASET_ROOT, "train", "labels")

VAL_IMAGES = os.path.join(DATASET_ROOT, "validation", "images")
VAL_LABELS = os.path.join(DATASET_ROOT, "validation", "labels")

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 1e-4


# -------------------------
# Dice Metric
# -------------------------
def dice_coefficient(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()


# -------------------------
# Training Loop
# -------------------------
def train():
    print(">>> Entered train()")
    print(">>> Using device:", DEVICE)

    # Dataset
    train_dataset = WoundSegmentationDataset(TRAIN_IMAGES, TRAIN_LABELS)
    val_dataset = WoundSegmentationDataset(VAL_IMAGES, VAL_LABELS)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # Windows-safe
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(">>> DataLoaders ready")
    print(">>> Train batches:", len(train_loader))
    print(">>> Val batches:", len(val_loader))

    # Model
    model = UNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_dice = 0.0

    print(">>> Starting training loop")

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        model.train()

        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 🔥 Heartbeat (first batch only)
            if batch_idx == 0:
                print("  ✔ First batch processed")

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                val_dice += dice_coefficient(outputs, masks).item()

        avg_val_dice = val_dice / len(val_loader)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f}"
        )

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "unet_baseline.pth"),
            )
            print("  💾 Best model saved")

    print("\n>>> Training completed")
    print(f">>> Best Validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    train()