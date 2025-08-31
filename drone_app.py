import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import base64

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Mareye Drone MVP", layout="wide")
st.title("🌊 Mareye Drone MVP: Pollution Region Highlighter")
st.markdown("""
Welcome to the Mareye Drone MVP demo!  
This tool highlights potential pollution regions (like oil spills) in marine images.  
You can upload your own images or try our **sample images** to see the detection in action.  

⚠️ **Note:** This is a **prototype**, so results may vary.  
""")

# ----------------------------
# Sidebar settings
# ----------------------------
st.sidebar.header("⚙️ Detection Settings")

# Default values
default_intensity = 100
default_canny_lower = 50
default_canny_upper = 150
default_kernel = 5
default_min_area = 100
default_alpha = 0.5
detection_modes = ["Basic", "Advanced"]

threshold_intensity = st.sidebar.slider("Intensity Threshold", 0, 255, default_intensity)
canny_lower = st.sidebar.slider("Canny Lower Threshold", 0, 255, default_canny_lower)
canny_upper = st.sidebar.slider("Canny Upper Threshold", 0, 255, default_canny_upper)
morph_kernel = st.sidebar.slider("Morph Kernel Size", 1, 21, default_kernel, step=2)
min_contour_area = st.sidebar.slider("Minimum Contour Area", 10, 1000, default_min_area)
overlay_alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, default_alpha, step=0.05)
detection_mode = st.sidebar.selectbox("Detection Mode", detection_modes)

st.sidebar.markdown("""
**Tips:**  
- Start with default values and upload an image.  
- Adjust **Intensity Threshold** to better reveal pollution regions.  
- **Canny thresholds** help detect edges (only used in Advanced mode).  
- **Morph Kernel Size** removes small noise.  
- **Minimum Contour Area** filters tiny irrelevant spots.  
- **Overlay Transparency** controls how strongly detected regions appear.  
- Try adjusting sliders slowly while observing the highlighted image.  
- Use "Advanced" mode to enable Canny-based detection for more detail.
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
# Sample Images Download (hyperlink style)
# ----------------------------
st.header("📂 Try Sample Images")
sample_zip_path = "sample_images.zip"
if os.path.exists(sample_zip_path):
    with open(sample_zip_path, "rb") as f:
        bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="sample_images.zip" style="text-decoration: underline; color: blue;">Download Sample Images Here</a>'
        st.markdown(href, unsafe_allow_html=True)

st.markdown("Download, unzip, and try the sample images above!")

# ----------------------------
# Upload user image
# ----------------------------
st.header("📤 Upload Your Own Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ----------------------------
    # Detection logic
    # ----------------------------
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Basic detection: original style
    _, mask = cv2.threshold(gray, threshold_intensity, 255, cv2.THRESH_BINARY_INV)

    # Morphology to clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Advanced detection: combine with Canny if selected
    if detection_mode == "Advanced":
        edges = cv2.Canny(gray, canny_lower, canny_upper)
        mask = cv2.bitwise_or(mask, edges)

    # Apply mask to original image
    result = img_rgb.copy()
    result[mask == 255] = [0, 0, 0]  # keep same style as processed_sample.jpg

    # Display side by side
    st.subheader("Detection Result")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.image(result, caption="Highlighted Pollution Regions", use_column_width=True)

    st.success("✅ Detection applied! Adjust the sliders to refine.")
