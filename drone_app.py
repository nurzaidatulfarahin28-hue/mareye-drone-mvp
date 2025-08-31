import streamlit as st
import zipfile
import base64
import os

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Mareye Drone MVP", layout="centered")

st.title("🌊 Mareye Drone MVP: Pollution Region Highlighter")
st.markdown("""
Welcome to the Mareye Drone MVP demo.  
This tool highlights potential pollution regions (e.g., oil spills) in marine images.  
Upload your own image or use our **sample images** to test.  

⚠️ **Note:** This is a **demo prototype**.  
In the future, Mareye will support **video streams, real-time GPS tracking, automated alerts, and dashboards**.
""")

# ----------------------------
# Instructions
# ----------------------------
st.header("📖 How to Use")
st.markdown("""
1. **Upload an image** (JPG or PNG).  
   👉 Example: aerial drone photo of the sea.  
2. **Adjust settings** (thresholds/sliders).  
   👉 Try tweaking until pollution regions are visible.  
3. **View results** with highlighted regions.  
4. Optionally, **download our sample images** below to test quickly.  
""")

# ⚠️ Judge Note
st.warning("""
⚠️ **Note for judges:**  
This demo is a **prototype**, so sometimes you may need to adjust the settings (sliders/thresholds)  
to see the pollution region highlighted correctly.  
The full version of Mareye will automate this process in real-time.
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
        btn = st.download_button(
            label="⬇️ Download Sample Images",
            data=f,
            file_name="sample_images.zip",
            mime="application/zip"
        )

st.markdown("""
Once downloaded, unzip the file and upload the samples above.  
You'll see how the system detects and highlights marine pollution areas.
""")

# ----------------------------
# Upload user image
# ----------------------------
st.header("📤 Upload Your Own Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.success("✅ Image uploaded successfully. (Pollution detection runs here in full version.)")

    # Placeholder for processing
    st.info("🔍 In a real system, this would highlight pollution areas automatically.")
