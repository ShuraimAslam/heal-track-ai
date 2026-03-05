def assess_wound_risk(metrics: dict):
    score = 0

    area_ratio = metrics["area_ratio"]
    shape_complexity = metrics["shape_complexity"]
    num_regions = metrics["num_regions"]
    confidence = metrics["segmentation_confidence"]

    # --- Area contribution ---
    if area_ratio > 0.05:
        score += 30
    elif area_ratio > 0.02:
        score += 20
    elif area_ratio > 0.005:
        score += 10

    # --- Shape complexity ---
    if shape_complexity > 3000:
        score += 20
    elif shape_complexity > 1500:
        score += 10

    # --- Fragmentation ---
    if num_regions >= 3:
        score += 20
    elif num_regions == 2:
        score += 10

    # --- Confidence penalty ---
    if confidence == "low":
        score += 25
    elif confidence == "none":
        score += 40

    # --- Initial risk level ---
    if score >= 70:
        risk_level = "High"
    elif score >= 35:
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
        "confidence": confidence
    }