import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Mareye Drone MVP", layout="wide")
st.title("🌊 Mareye Drone MVP: Pollution Region Highlighter")
st.markdown("""
Welcome to the Mareye Drone MVP demo.  
This tool highlights potential pollution regions (e.g., oil spills) in marine images.  
Upload your own image or use **sample images** to test.  

⚠️ **Note:** This is a **demo prototype**.
""")

# ----------------------------
# Sidebar settings (detection parameters)
# ----------------------------
st.sidebar.header("⚙️ Detection Settings")

# Default values
default_intensity = 100
default_canny = 50
default_kernel = 5
default_min_area = 100
default_alpha = 0.5

threshold_intensity = st.sidebar.slider("Intensity Threshold", 0, 255, default_intensity)
canny_lower = st.sidebar.slider("Canny Lower Threshold", 0, 255, default_canny)
morph_kernel = st.sidebar.slider("Morph Kernel Size", 1, 21, default_kernel, step=2)
min_contour_area = st.sidebar.slider("Minimum Contour Area", 10, 1000, default_min_area)
overlay_alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, default_alpha, step=0.05)

st.sidebar.markdown("""
**Tips:**  
- Default values are set for demonstration.  
- Judges should carefully adjust sliders to refine detection for each image.  
- Try increasing intensity threshold for darker pollution areas.  
- Adjust Canny lower threshold for edge detection sensitivity.  
- Morph kernel can help remove noise.  
- Minimum contour area filters small false positives.  
- Overlay transparency controls how strongly detected regions appear.
""")

# ----------------------------
# Example Section (Before/After demo)
# ----------------------------
st.header("🖼️ Example Output")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image("examples/original_sample.jpg", caption="Before - Marine Image")

with col2:
    st.subheader("Highlighted Result")
    st.image("examples/processed_sample.jpg", caption="After - Pollution Highlighted")

# ----------------------------
# Sample Images Download (prominently visible)
# ----------------------------
st.header("📂 Download Sample Images")
sample_zip_path = "sample_images.zip"

if os.path.exists(sample_zip_path):
    with open(sample_zip_path, "rb") as f:
        st.download_button(
            label="⬇️ Click to Download Sample Images",
            data=f,
            file_name="sample_images.zip",
            mime="application/zip",
            key="download_samples"
        )
st.markdown("Download, unzip, and upload the sample images above to test detection.")

# ----------------------------
# Upload user image
# ----------------------------
st.header("📤 Upload Your Own Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    img_pil = Image.open(uploaded_file)
    img = np.array(img_pil)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ----------------------------
    # Detection logic
    # ----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply intensity threshold
    _, mask = cv2.threshold(gray, threshold_intensity, 255, cv2.THRESH_BINARY_INV)

    # Optional: Morphology to clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create overlay
    overlay = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_contour_area:
            cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), -1)  # filled red

    # Blend overlay with original
    result = cv2.addWeighted(overlay, overlay_alpha, img, 1 - overlay_alpha, 0)

    st.image(result, caption="Highlighted Pollution Regions", use_column_width=True)
    st.success("✅ Detection applied! Adjust the sliders to refine.")
