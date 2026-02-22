import os
import cv2
import torch
from torch.utils.data import Dataset


class WoundSegmentationDataset(Dataset):
    """
    PyTorch Dataset for wound segmentation training.

    Used ONLY for training / validation.
    """

    def __init__(self, images_dir, labels_dir, image_size=(256, 256)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size

        self.image_files = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name)

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image.astype("float32") / 255.0

        # Load mask
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load label: {label_path}")

        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype("float32")

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)        # (1, H, W)

        return image, mask