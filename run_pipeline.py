import sys
from src.segmentation.inference import segment_wound
from src.measurement.wound_metrics import compute_wound_metrics
from src.reasoning.clinical_rules import assess_wound_risk
from src.insight.report import generate_clinical_report

IMAGE_PATH = "sample_data/wound.png" # change as needed



def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/wound.png"
    mask = segment_wound(image_path)
    metrics = compute_wound_metrics(mask)
    decision = assess_wound_risk(metrics)
    report = generate_clinical_report(metrics, decision)
    print(report)

if __name__ == "__main__":
    main()