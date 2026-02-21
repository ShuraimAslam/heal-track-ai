def assess_wound_size(area_ratio: float) -> str:
    """
    Categorize wound size based on area ratio.
    """
    if area_ratio < 0.05:
        return "small"
    elif area_ratio < 0.15:
        return "medium"
    else:
        return "large"


def assess_risk_level(size_category: str) -> str:
    """
    Map wound size to clinical risk level.
    """
    if size_category == "small":
        return "low"
    elif size_category == "medium":
        return "medium"
    else:
        return "high"


def assess_healing_status(size_category: str) -> str:
    """
    Estimate healing status based on wound size.
    """
    if size_category == "small":
        return "improving"
    elif size_category == "medium":
        return "stable"
    else:
        return "concerning"


def clinical_reasoning(area_ratio: float) -> dict:
    """
    Main reasoning function.
    Converts measurement into clinical interpretation.
    """
    size_category = assess_wound_size(area_ratio)
    risk_level = assess_risk_level(size_category)
    healing_status = assess_healing_status(size_category)

    recommend_review = True if risk_level == "high" else False

    reason_summary = f"Wound classified as {size_category} based on area ratio"

    return {
        "risk_level": risk_level,
        "healing_status": healing_status,
        "recommend_clinician_review": recommend_review,
        "reason_summary": reason_summary
    }