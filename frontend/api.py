import sys
import os
import io
import base64
import uuid
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Add project root to path so we can import from src
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from src.segmentation.inference import segment_wound
from src.measurement.wound_metrics import compute_wound_metrics
from src.reasoning.clinical_rules import assess_wound_risk
from src.insight.report import generate_clinical_report

app = FastAPI(title="Heal-Track API")

# Enable CORS for Next.js development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def to_uint8(mask):
    """Convert float mask [0, 1] to uint8 [0, 255] for visualization."""
    return (mask * 255).astype(np.uint8)

def create_overlay(image_pil, mask_np):
    """Creates a transparent overlay of the mask on the image."""
    img = np.array(image_pil)
    if mask_np.shape[:2] != img.shape[:2]:
        mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask_np

    overlay = img.copy()
    overlay[mask_resized > 0.5] = [255, 0, 0]
    combined = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    return Image.fromarray(combined)

def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Heal-Track API is running"}

@app.post("/analyze")
async def analyze_wound(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 1. Segmentation
        mask = segment_wound(input_image)
        
        # 2. Metrics
        metrics = compute_wound_metrics(mask)
        
        # 3. Clinical Risk
        risk = assess_wound_risk(metrics)
        
        # 4. Report
        report = generate_clinical_report(metrics, risk)
        
        # 5. Visualizations
        overlay_image = create_overlay(input_image, mask)
        mask_vis = Image.fromarray(to_uint8(mask))
        
        # Prepare response
        return {
            "case_id": str(uuid.uuid4())[:8],
            "metrics": metrics,
            "risk": risk,
            "report": report,
            "visualizations": {
                "overlay": f"data:image/png;base64,{pil_to_base64(overlay_image)}",
                "mask": f"data:image/png;base64,{pil_to_base64(mask_vis)}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
