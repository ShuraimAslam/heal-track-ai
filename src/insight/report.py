def generate_clinical_report(metrics: dict, risk: dict):
    confidence = risk["confidence"]
    risk_level = risk["risk_level"]

    lines = []

    lines.append("AI-Assisted Wound Analysis Summary")
    lines.append("-" * 40)

    lines.append(f"Wound Area (pixels): {metrics['wound_area']}")
    lines.append(f"Area Ratio: {metrics['area_ratio']}")
    lines.append(f"Detected Regions: {metrics['num_regions']}")
    lines.append(f"Segmentation Confidence: {confidence.capitalize()}")

    lines.append("\nRisk Assessment:")
    lines.append(f"Overall Risk Level: {risk_level}")
    lines.append(f"Risk Score: {risk['risk_score']} / 100")

    lines.append("\nInterpretation:")

    if confidence == "high":
        lines.append(
            "The segmentation quality is adequate, and the risk assessment "
            "is based on geometric and structural characteristics of the wound."
        )
    else:
        lines.append(
            "The system detected a wound region, but confidence in the "
            "segmentation is limited. Risk assessment has been escalated "
            "to avoid underestimation."
        )

    if risk_level in ["Moderate", "High"]:
        lines.append(
            "\nRecommendation: Clinical review is advised for further evaluation."
        )
    else:
        lines.append(
            "\nRecommendation: Continue monitoring. This assessment is "
            "intended for decision support only."
        )

    lines.append(
        "\nDisclaimer: This is a research prototype and should not be used "
        "for medical diagnosis or treatment decisions."
    )

    return "\n".join(lines)