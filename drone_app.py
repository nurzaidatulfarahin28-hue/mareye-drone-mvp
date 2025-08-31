import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import base64

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Mareye Drone MVP", layout="wide")
st.title("🌊 Mareye Drone MVP: Oil & Debris Detector")
st.markdown("""
This tool highlights potential **oil spills** and **floating debris** in marine images.  
Upload an aerial photo, adjust detection settings, and see the results in real-time.
""")

# ----------------------------
# Sidebar Settings
# ----------------------------
st.sidebar.header("⚙️ Detection Settings")

default_intensity = 100
default_kernel = 5
default_min_area = 100
default_alpha = 0.5
detection_modes = ["Oil", "Debris", "Both"]

threshold_intensity = st.sidebar.slider("Intensity Threshold", 0, 255, default_intensity)
morph_kernel = st.sidebar.slider("Morph Kernel Size", 1, 21, default_kernel, step=2)
min_contour_area = st.sidebar.slider("Minimum Contour Area", 10, 1000, default_min_area)
overlay_alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, default_alpha, step=0.05)
detection_mode = st.sidebar.selectbox("Detection Mode", detection_modes)

st.sidebar.markdown("""
- **Oil**: Detects dark, smooth patches.  
- **Debris**: Detects bright or irregular objects.  
- **Both**: Combines detection for oil & debris.  
""")

oil_color = (0, 255, 0)    # Green
debris_color = (0, 0, 255) # Red

# ----------------------------
# Example Section
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
# Sample Images Download
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
# Upload User Image
# ----------------------------
st.header("📤 Upload Your Own Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))

    # Detection masks
    oil_mask, debris_mask = None, None
    if detection_mode in ["Oil", "Both"]:
        _, oil_mask = cv2.threshold(gray, threshold_intensity, 255, cv2.THRESH_BINARY_INV)
        oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_CLOSE, kernel)

    if detection_mode in ["Debris", "Both"]:
        _, debris_mask = cv2.threshold(gray, threshold_intensity, 255, cv2.THRESH_BINARY)
        debris_mask = cv2.morphologyEx(debris_mask, cv2.MORPH_CLOSE, kernel)

    # Combine masks if needed
    if detection_mode == "Both":
        mask = cv2.bitwise_or(oil_mask, debris_mask)
    elif detection_mode == "Oil":
        mask = oil_mask
    else:
        mask = debris_mask

    # Contour detection & overlay
    overlay = img_rgb.copy()
    if detection_mode in ["Oil", "Both"] and oil_mask is not None:
        contours, _ = cv2.findContours(oil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_contour_area:
                cv2.drawContours(overlay, [cnt], -1, oil_color, 2)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(overlay, (x, y), (x+w, y+h), oil_color, 2)

    if detection_mode in ["Debris", "Both"] and debris_mask is not None:
        contours, _ = cv2.findContours(debris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_contour_area:
                cv2.drawContours(overlay, [cnt], -1, debris_color, 2)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(overlay, (x, y), (x+w, y+h), debris_color, 2)

    # Blend overlay
    result = cv2.addWeighted(overlay, overlay_alpha, img_rgb, 1 - overlay_alpha, 0)

    # Display results
    st.subheader("Detection Result")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.image(result, caption=f"Detected {detection_mode}", use_column_width=True)

    # Save option
    if st.button("💾 Save Processed Image"):
        out_path = "highlighted_result.png"
        cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="highlighted_result.png">Download Here</a>'
            st.markdown(href, unsafe_allow_html=True)

    st.success("✅ Detection complete! Adjust sliders for best results.")
