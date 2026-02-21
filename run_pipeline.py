from src.segmentation.model import SegmentationModel
from src.measurement.wound_metrics import (
    compute_pixel_area,
    compute_area_ratio,
    compute_bounding_box
)
from src.reasoning.clinical_rules import clinical_reasoning
from src.insight.report import generate_insight


def main():
    image_path = "image.png"

    # 1. Segmentation (placeholder)
    segmenter = SegmentationModel()
    mask = segmenter.predict(image_path)

    # 2. Measurement
    pixel_area = compute_pixel_area(mask)
    area_ratio = compute_area_ratio(mask)
    bounding_box = compute_bounding_box(mask)

    measurements = {
        "pixel_area": pixel_area,
        "area_ratio": area_ratio,
        "bounding_box": bounding_box
    }

    # 3. Reasoning
    reasoning_output = clinical_reasoning(area_ratio)

    # 4. Insight
    insight_text = generate_insight(reasoning_output)

    # 5. Display results
    print("\n--- PIPELINE OUTPUT ---")
    print("Measurements:", measurements)
    print("Reasoning:", reasoning_output)
    print("Insight:", insight_text)
    print("----------------------\n")


if __name__ == "__main__":
    main()