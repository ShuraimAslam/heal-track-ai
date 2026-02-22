def assess_wound_size(metrics: dict) -> str:
    """
    Categorize wound size based on area ratio.
    """
    area_ratio = metrics["area_ratio"]

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


def assess_wound_risk(metrics: dict) -> dict:
    """
    Main clinical reasoning entry point for the pipeline.
    """
    size_category = assess_wound_size(metrics)
    risk_level = assess_risk_level(size_category)
    healing_status = assess_healing_status(size_category)

    recommend_review = risk_level in ("medium", "high")

    return {
        "risk_level": risk_level.capitalize(),
        "healing_status": healing_status,
        "recommend_clinician_review": recommend_review,
        "size_category": size_category
    }