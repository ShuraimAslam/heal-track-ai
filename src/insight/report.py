def generate_insight(reasoning_output: dict) -> str:
    """
    Convert reasoning output into a human-readable clinical insight.
    """

    risk = reasoning_output.get("risk_level")
    status = reasoning_output.get("healing_status")
    recommend = reasoning_output.get("recommend_clinician_review")

    # Base message
    if status == "improving":
        message = "The wound appears to be healing well."
    elif status == "stable":
        message = "The wound condition appears stable at this time."
    else:
        message = "The wound appears concerning and requires attention."

    # Risk-based augmentation
    if risk == "medium":
        message += " Continued monitoring is advised."
    elif risk == "high":
        message += " The wound is classified as high risk."

    # Action recommendation
    if recommend:
        message += " A clinical review is recommended."

    return message