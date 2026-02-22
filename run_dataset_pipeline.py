from src.segmentation.dataset_adapter import DatasetAdapter
from src.segmentation.inference import SegmentationInference

from src.measurement.wound_metrics import (
    compute_pixel_area,
    compute_area_ratio,
    compute_bounding_box
)

from src.reasoning.clinical_rules import clinical_reasoning
from src.insight.report import generate_insight


def main():
    # 1. Dataset adapter
    adapter = DatasetAdapter(
        dataset_root="D:/AIML/dataset/Foot Ulcer Segmentation Challenge"
    )

    # 2. Segmentation inference
    segmenter = SegmentationInference()

    # 3. Select small subset
    items = adapter.get_items("train")[:5]

    for idx, (image_path, _) in enumerate(items):
        print(f"\n=== SAMPLE {idx + 1} ===")
        print(f"Image: {image_path}")

        # Segmentation
        mask = segmenter.predict(image_path)

        # Measurement
        pixel_area = compute_pixel_area(mask)
        area_ratio = compute_area_ratio(mask)
        bounding_box = compute_bounding_box(mask)

        measurements = {
            "pixel_area": pixel_area,
            "area_ratio": area_ratio,
            "bounding_box": bounding_box,
        }

        # Reasoning
        reasoning_output = clinical_reasoning(area_ratio)

        # Insight
        insight_text = generate_insight(reasoning_output)

        # Output
        print("Measurements:", measurements)
        print("Reasoning:", reasoning_output)
        print("Insight:", insight_text)


if __name__ == "__main__":
    main()