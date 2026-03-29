def assess_wound_risk(metrics: dict):
    score = 0

    area_ratio = metrics["area_ratio"]
    shape_complexity = metrics["shape_complexity"]
    num_regions = metrics["num_regions"]
    confidence = metrics["segmentation_confidence"]

    # --- Area contribution ---
    if area_ratio >= 0.03:   # >= 3% of image is considered severe
        score += 50
    elif area_ratio >= 0.01: # >= 1% is moderate
        score += 30
    elif area_ratio >= 0.002: # small wound
        score += 15

    # --- Shape complexity ---
    # A perfect circle has complexity ~12.5. Highly irregular shapes indicate higher risk.
    if shape_complexity >= 40:
        score += 20
    elif shape_complexity >= 20:
        score += 10

    # --- Fragmentation ---
    if num_regions >= 3:
        score += 20
    elif num_regions == 2:
        score += 10

    # --- Initial risk level ---
    if score >= 50:
        risk_level = "High"
    elif score >= 25:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # --- CRITICAL SAFETY RULES ---
    # 1. If a wound exists, LOW risk is not allowed
    if area_ratio > 0.005 and risk_level == "Low":
        risk_level = "Moderate"

    # 2. If confidence is low/none, never allow LOW
    if confidence in ["low", "none"] and risk_level == "Low":
        risk_level = "Moderate"

    return {
        "risk_level": risk_level,
        "risk_score": score,
        "confidence": confidence,
        "needs_clinical_review": confidence in ["low", "none"]
    }