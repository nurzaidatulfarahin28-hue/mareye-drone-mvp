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
Upload your own image or use our **sample images** to test.  

⚠️ **Note:** This is a **demo prototype**.  
In the future, Mareye will support **video streams, real-time GPS tracking, automated alerts, and dashboards**.
""")

# ----------------------------
# Sidebar settings
# ----------------------------
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Pollution Detection Threshold", 0, 255, 100)
min_size = st.sidebar.slider("Minimum Region Size (pixels)", 20, 100, 30)

# ----------------------------
# Instructions
# ----------------------------
st.header("📖 How to Use")
st.markdown("""
1. **Upload an image** (JPG or PNG).  
2. Adjust **detection settings** using the sidebar sliders.  
3. View detected regions highlighted on the image.  
4. Optionally, **download our sample images** to test quickly.  
""")

# ----------------------------
# Example Section (Before/After)
# ----------------------------
st.header("🖼️ Example Output")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image("examples/original_sample.jpg", caption="Before - Marine Image")

with col2:
    st.subheader("Detected Regions")
    img = cv2.imread("examples/original_sample.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img_rgb.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_size and h >= min_size:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green boxes

    st.image(output, caption="After - Pollution Highlighted", use_column_width=True)

# ----------------------------
# Sample Images Download
# ----------------------------
st.header("📂 Try It Yourself")
sample_zip_path = "sample_images.zip"

if os.path.exists(sample_zip_path):
    with open(sample_zip_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Sample Images",
            data=f,
            file_name="sample_images.zip",
            mime="application/zip"
        )

st.markdown("Once downloaded, unzip the file and upload the samples above.")

# ----------------------------
# Upload user image
# ----------------------------
st.header("📤 Upload Your Own Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file)
    img = np.array(img_pil)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_size and h >= min_size:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green boxes

    st.image(output, caption="Highlighted Pollution Regions", use_column_width=True)
    st.success("✅ Detection applied! Adjust the sliders to refine.")
