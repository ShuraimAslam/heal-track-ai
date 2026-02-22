import cv2
import numpy as np


class SegmentationInference:
    """
    Segmentation inference wrapper.

    Responsibility:
    - Load image from path
    - Preprocess image
    - Run segmentation model (placeholder for now)
    - Return binary mask
    """

    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size
        # NOTE: real model will be loaded here later

    def preprocess(self, image):
        """
        Resize and normalize image.
        """
        image = cv2.resize(image, self.image_size)
        image = image.astype("float32") / 255.0
        return image

    def predict(self, image_path):
        """
        Run segmentation inference on a single image.

        Returns:
        - binary mask (numpy array, 0/1)
        """
        # 1. Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Preprocess
        image = self.preprocess(image)

        # 3. Placeholder segmentation logic
        # (simple intensity threshold to simulate a model)
        gray = cv2.cvtColor((image * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 40, 1, cv2.THRESH_BINARY)

        return mask