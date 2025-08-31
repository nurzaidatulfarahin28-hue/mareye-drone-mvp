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
# Sidebar settings
# ----------------------------
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Pollution Detection Threshold", 0, 255, 100)

# ----------------------------
# Instructions
# ----------------------------
st.header("📖 How to Use")
st.markdown("""
1. Upload an image (JPG or PNG).  
2. Adjust the **detection threshold** in the sidebar.  
3. View results with highlighted pollution regions.  
4. Optionally, download our **sample images** to test.
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
    # Read uploaded image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # ----------------------------
    # Original detection logic
    # ----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    result = img_rgb.copy()
    result[mask == 255] = [0, 0, 0]  # keep the original detection logic (dark regions highlighted)

    st.image(result, caption="Highlighted Pollution Regions", use_column_width=True)
    st.success("✅ Detection applied! Adjust the slider to refine.")
