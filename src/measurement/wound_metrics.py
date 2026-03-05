import numpy as np
import cv2

def compute_wound_metrics(mask: np.ndarray):
    """
    mask: binary mask (0 or 1)
    """

    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    image_area = h * w

    # --- Area ---
    wound_area = int(mask.sum())
    area_ratio = wound_area / image_area if image_area > 0 else 0

    # --- Contours ---
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    num_regions = len(contours)

    perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)

    # --- Shape complexity ---
    shape_complexity = (
        (perimeter ** 2) / (wound_area + 1e-6)
        if wound_area > 0 else 0
    )

    # --- Segmentation confidence (heuristic) ---
    if wound_area == 0:
        confidence = "none"
    elif area_ratio < 0.005:
        confidence = "low"
    elif area_ratio < 0.02:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        "wound_area": wound_area,
        "area_ratio": round(area_ratio, 4),
        "shape_complexity": round(shape_complexity, 2),
        "num_regions": num_regions,
        "segmentation_confidence": confidence
    }