import cv2
import numpy as np

class SegmentationModel:
    """
    Placeholder segmentation model (Phase 3A).
    This will later be replaced by a trained UNet.
    """

    def __init__(self, threshold: int = 120):
        self.threshold = threshold

    def predict(self, image_path: str) -> np.ndarray:
        """
        Takes an image path and returns a binary mask.
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")

        # Resize to standard size (must match dataset contract)
        image = cv2.resize(image, (256, 256))

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Simple thresholding as placeholder segmentation
        _, mask = cv2.threshold(
            gray, self.threshold, 1, cv2.THRESH_BINARY
        )

        return mask.astype(np.uint8)