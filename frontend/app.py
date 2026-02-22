import streamlit as st
import numpy as np
import cv2
import os
import time
import json
import uuid
from PIL import Image
import sys
from pathlib import Path

# Add project root to path so we can import from src
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from src.segmentation.inference import segment_wound
from src.measurement.wound_metrics import compute_wound_metrics
from src.reasoning.clinical_rules import assess_wound_risk
from src.insight.report import generate_clinical_report

# --- UI CONSTANTS & CONFIG ---
st.set_page_config(
    page_title="Heal-Track | AI Wound Analysis",
    page_icon="🩹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .report-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #6c757d;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

def to_uint8(mask):
    """Convert float mask [0, 1] to uint8 [0, 255] for visualization."""
    return (mask * 255).astype(np.uint8)

def create_overlay(image_pil, mask_np):
    """Creates a transparent overlay of the mask on the image."""
    img = np.array(image_pil)
    # Resize mask to match image size if necessary
    if mask_np.shape[:2] != img.shape[:2]:
        mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask_np

    # Create solid color overlay (red)
    overlay = img.copy()
    overlay[mask_resized > 0.5] = [255, 0, 0]
    
    # Blend with original for transparency
    combined = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    return Image.fromarray(combined)

def save_case_results(case_id, image, mask, overlay, metrics, risk, report):
    """Saves the results of the analysis to a case directory."""
    case_dir = root_path / "outputs" / f"case_{case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Save input image
    image.save(case_dir / "input_image.png")
    
    # Handle mask using helper
    mask_vis = to_uint8(mask)
    Image.fromarray(mask_vis).save(case_dir / "predicted_mask.png")
    
    overlay.save(case_dir / "overlay_image.png")
    
    # Save report
    with open(case_dir / "report.txt", "w") as f:
        f.write(report)
    
    # Save metrics and risk logic
    results = {
        "metrics": metrics,
        "risk": risk
    }
    with open(case_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return case_dir

# --- HEADER ---
header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    st.title("🩹")
with header_col2:
    st.title("Heal-Track")
    st.subheader("AI-Assisted Wound Analysis Prototype")

st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    uploaded_file = st.file_uploader("Upload Wound Image", type=["jpg", "png", "jpeg"])
    
    case_id = str(uuid.uuid4())[:8]
    st.info(f"Session Case ID: **{case_id}**")
    
    run_analysis = st.button("🚀 Run Analysis", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool uses a U-Net model for segmentation and rule-based logic for clinical assessment.
    
    Developed for research and demonstration purposes.
    """)

# --- MAIN CONTENT ---
if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    
    if run_analysis:
        with st.status("Analyzing wound...", expanded=True) as status:
            st.write("Performing segmentation...")
            mask = segment_wound(input_image)
            
            st.write("Calculating metrics...")
            metrics = compute_wound_metrics(mask)
            
            st.write("Assessing clinical risk...")
            risk = assess_wound_risk(metrics)
            
            st.write("Generating report...")
            report = generate_clinical_report(metrics, risk)
            
            st.write("Creating visualization...")
            overlay_image = create_overlay(input_image, mask)
            
            st.write("Saving case file...")
            case_path = save_case_results(case_id, input_image, mask, overlay_image, metrics, risk, report)
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
        
        st.success(f"Results saved to: `{case_path.name}`")
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### A. Input Preview")
            st.image(input_image, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            st.markdown("### B. Segmentation Results")
            tab1, tab2 = st.tabs(["Overlay View", "Raw Mask"])
            with tab1:
                st.image(overlay_image, caption="Mask Overlay", use_container_width=True)
            with tab2:
                # Use helper for UI consistency
                st.image(to_uint8(mask), caption="Predicted Mask", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### C. Wound Metrics Dashboard")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Pixel Area", f"{metrics['pixel_area']} px")
        m_col2.metric("Area Ratio", f"{metrics['area_ratio']*100:.2f}%")
        m_col3.metric("Risk Level", risk['risk_level'])
        
        st.markdown("---")
        st.markdown("### D. Clinical Summary")
        
        # Styling based on risk
        alert_type = st.info
        if risk['risk_level'].lower() == 'high':
            alert_type = st.error
        elif risk['risk_level'].lower() == 'medium':
            alert_type = st.warning
            
        alert_type(f"**Insight:** {report.split('Insight:')[-1].strip()}")
        
        with st.expander("View Full Report"):
            st.text(report)

    else:
        st.markdown("### A. Input Preview")
        st.image(input_image, caption="Pending Analysis", use_container_width=True)
        st.info("Click 'Run Analysis' in the sidebar to start.")

else:
    st.info("Please upload a wound image in the sidebar to begin.")
    
    # Showcase placeholders
    st.markdown("### How it works")
    st.image("https://via.placeholder.com/800x200.png?text=Upload+an+Image+to+See+the+AI+Workflow", use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <strong>⚠️ Safety Disclaimer</strong><br>
        This is a research prototype. Results are AI-generated and should not be used for medical decisions.<br>
        Heal-Track © 2026 AI Wound Analysis Demo.
    </div>
    """, 
    unsafe_allow_html=True
)
