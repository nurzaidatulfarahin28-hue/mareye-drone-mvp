import streamlit as st
import cv2
import numpy as np
import os

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
# Instructions
# ----------------------------
st.header("📖 How to Use")
st.markdown("""
1. Upload an image (JPG or PNG).  
2. Adjust the **detection threshold** to highlight pollution regions.  
3. View results with highlighted regions.  
4. Optionally, download sample images to test.
""")

# ----------------------------
# Sidebar settings
# ----------------------------
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Pollution Detection Threshold", 0, 255, 100)

# ----------------------------
# Example Section (Before/After)
# ----------------------------
st.header("🖼️ Example Output")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image("examples/original_sample.jpg", caption="Before - Marine Image", use_column_width=True)

with col2:
    st.subheader("Highlighted Result")
    img = cv2.imread("examples/original_sample.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Simple detection: highlight dark regions as "pollution"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Create semi-transparent overlay
    mask_color = np.zeros_like(img_rgb)
    mask_color[mask == 255] = [255, 0, 0]  # red for highlight
    alpha = 0.3  # transparency
    result = cv2.addWeighted(img_rgb, 1, mask_color, alpha, 0)
    
    st.image(result, caption="After - Pollution Highlighted", use_column_width=True)

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

# ----------------------------
# Upload user image
# ----------------------------
st.header("📤 Upload Your Own Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # Detection on uploaded image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    mask_color = np.zeros_like(img_rgb)
    mask_color[mask == 255] = [255, 0, 0]  # semi-transparent red overlay
    alpha = 0.3
    result = cv2.addWeighted(img_rgb, 1, mask_color, alpha, 0)
    
    st.image(result, caption="Highlighted Pollution Regions", use_column_width=True)
    st.success("✅ Detection applied! Adjust the slider to refine.")
