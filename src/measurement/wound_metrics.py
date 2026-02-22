import numpy as np
import cv2

def compute_pixel_area(mask: np.ndarray) -> int:
    """
    Counts number of wound pixels (mask == 1)
    """
    return int(np.sum(mask))

def compute_area_ratio(mask: np.ndarray) -> float:
    """
    Ratio of wound pixels to total image pixels
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    wound_pixels = np.sum(mask)
    return float(wound_pixels / total_pixels)

def compute_bounding_box(mask: np.ndarray):
    """
    Returns bounding box of wound as (x_min, y_min, x_max, y_max)
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return int(x_min), int(y_min), int(x_max), int(y_max)

def compute_wound_metrics(mask: np.ndarray) -> dict:
        """
    Aggregates all wound measurements into a single dictionary
    """
        pixel_area = compute_pixel_area(mask)
        area_ratio = compute_area_ratio(mask)
        bounding_box = compute_bounding_box(mask)

        return {
        "pixel_area": pixel_area,
        "area_ratio": round(area_ratio, 4),
        "bounding_box": bounding_box
    }