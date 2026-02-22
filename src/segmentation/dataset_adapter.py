import os
import cv2


class DatasetAdapter:
    """
    Adapter for Foot Ulcer Segmentation Challenge dataset.

    Responsibilities:
    - Understand dataset structure
    - Return image/label paths per split
    - Load images and labels consistently
    """

    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root

        self.splits = {
            "train": {
                "images": os.path.join(dataset_root, "train", "images"),
                "labels": os.path.join(dataset_root, "train", "labels"),
                "has_labels": True,
            },
            "validation": {
                "images": os.path.join(dataset_root, "validation", "images"),
                "labels": os.path.join(dataset_root, "validation", "labels"),
                "has_labels": True,
            },
            "test": {
                "images": os.path.join(dataset_root, "test", "images"),
                "labels": None,
                "has_labels": False,
            },
        }

        self._validate_structure()

    def _validate_structure(self):
        for split, cfg in self.splits.items():
            if not os.path.isdir(cfg["images"]):
                raise FileNotFoundError(f"Missing images directory for split: {split}")

            if cfg["has_labels"] and not os.path.isdir(cfg["labels"]):
                raise FileNotFoundError(f"Missing labels directory for split: {split}")

    def get_items(self, split: str):
        """
        Returns list of (image_path, label_path or None)
        """
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}")

        cfg = self.splits[split]
        image_files = sorted(os.listdir(cfg["images"]))

        items = []

        for img_name in image_files:
            img_path = os.path.join(cfg["images"], img_name)

            if cfg["has_labels"]:
                label_path = os.path.join(cfg["labels"], img_name)
                if not os.path.exists(label_path):
                    raise FileNotFoundError(
                        f"Label missing for image {img_name} in split {split}"
                    )
            else:
                label_path = None

            items.append((img_path, label_path))

        return items

    @staticmethod
    def load_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def load_label(label_path):
        if label_path is None:
            return None

        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load label: {label_path}")

        # Ensure binary mask
        mask = (mask > 0).astype("uint8")
        return mask