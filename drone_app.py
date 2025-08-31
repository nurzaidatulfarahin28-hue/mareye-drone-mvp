import io
from typing import Tuple, Dict

import numpy as np
from PIL import Image
import cv2
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Mareye Drone MVP (Image Detector)",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Mareye Drone MVP — Pollution-Like Region Highlighter")
st.caption("Upload an aerial image. The app highlights dark, oil-like patches or high-contrast debris using basic OpenCV.")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Detection Settings")

mode = st.sidebar.selectbox(
    "Detection mode",
    ["Dark patches (oil-like)", "High-contrast debris"],
    help="Pick a simple heuristic suited to your image type."
)

threshold = st.sidebar.slider(
    "Intensity threshold (lower = darker)",
    min_value=0, max_value=255, value=70, step=1,
    help="Used in 'Dark patches' mode on the grayscale image."
)

edge_low = st.sidebar.slider(
    "Canny lower threshold", 0, 255, 60, 1,
    help="Used in 'High-contrast debris' mode."
)
edge_high = st.sidebar.slider(
    "Canny upper threshold", 0, 255, 140, 1,
    help="Used in 'High-contrast debris' mode."
)

kernel_size = st.sidebar.slider(
    "Morph kernel (odd)", 1, 19, 7, 2,
    help="Size of morphological kernel for noise removal / region closing."
)
min_area = st.sidebar.slider(
    "Min contour area (px)", 0, 20000, 800, 50,
    help="Rejects tiny detections."
)
overlay_alpha = st.sidebar.slider(
    "Overlay transparency", 0.0, 1.0, 0.35, 0.05,
    help="How strong the colored overlay looks."
)

show_masks = st.sidebar.checkbox("Show intermediate mask/edges", value=False)
st.sidebar.markdown("---")
st.sidebar.info("Tip: start with the default settings, then tweak threshold/area only if needed.")

# -----------------------------
# Helpers
# -----------------------------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def create_kernel(k: int) -> np.ndarray:
    k = max(1, k)
    if k % 2 == 0:
        k += 1
    return np.ones((k, k), np.uint8)

def dark_patch_mask(bgr: np.ndarray, thr: int, ksize: int) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    kernel = create_kernel(ksize)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def debris_mask(bgr: np.ndarray, low: int, high: int, ksize: int) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    kernel = create_kernel(ksize)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # Convert edges to filled regions by finding contours & filling
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def annotate(bgr: np.ndarray, mask: np.ndarray, min_area_px: int, alpha: float) -> Tuple[np.ndarray, Dict[str, float]]:
    h, w = mask.shape[:2]
    overlay = bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kept = []
    total_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area_px:
            kept.append(c)
            total_area += area

    # colored overlay (blue-ish)
    color = (255, 80, 0)  # BGR (teal-ish)
    mask_color = np.zeros_like(bgr)
    cv2.drawContours(mask_color, kept, -1, color, thickness=cv2.FILLED)
    overlay = cv2.addWeighted(overlay, 1.0, mask_color, alpha, 0)

    # outlines + boxes
    cv2.drawContours(overlay, kept, -1, (255, 255, 255), thickness=2)
    for c in kept:
        x, y, ww, hh = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x+ww, y+hh), (0, 0, 0), 1)

    coverage_pct = 100.0 * total_area / float(h * w)
    stats = {
        "regions": len(kept),
        "coverage_percent": coverage_pct,
        "image_width": w,
        "image_height": h
    }
    return overlay, stats

def process_image(pil_img: Image.Image) -> Tuple[Image.Image, Dict[str, float], np.ndarray]:
    bgr = pil_to_cv(pil_img)
    if mode == "Dark patches (oil-like)":
        mask = dark_patch_mask(bgr, threshold, kernel_size)
    else:
        mask = debris_mask(bgr, edge_low, edge_high, kernel_size)

    annotated, stats = annotate(bgr, mask, min_area, overlay_alpha)
    return cv_to_pil(annotated), stats, mask


# -----------------------------
# UI
# -----------------------------
uploaded = st.file_uploader("Upload an aerial image (JPG/PNG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
with col2:
    st.subheader("Annotated")

if uploaded is None:
    st.info("Drag & drop an image above to begin. For best results, use top-down aerial shots where pollution appears darker or high-contrast.")
else:
    image_bytes = uploaded.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    col1.image(pil_img, use_column_width=True)
    annotated_pil, stats, mask = process_image(pil_img)
    col2.image(annotated_pil, use_column_width=True)

    # Stats
    st.markdown("### Detection Summary")
    st.write(
        f"- Regions detected: **{int(stats['regions'])}**\n"
        f"- Estimated coverage: **{stats['coverage_percent']:.2f}%** of the image area\n"
        f"- Resolution: **{stats['image_width']}×{stats['image_height']}**"
    )

    # Optional debug
    if show_masks:
        st.markdown("### Intermediate Mask / Edges")
        st.image(mask, caption="Binary mask/edges", use_column_width=True)

    # Download
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download annotated image",
        data=buf.getvalue(),
        file_name="mareye_annotated.png",
        mime="image/png"
    )

st.markdown("---")
st.markdown("**Note:** This is a simple heuristic MVP. Future work: YOLO/Segmentation on real drone frames, edge AI, and telemetry integration.")