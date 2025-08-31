<<<<<<< HEAD
# mareye-drone-mvp
=======
# Mareye Drone MVP (Streamlit + OpenCV)

A minimal, ready-to-deploy MVP that simulates a drone by processing **uploaded aerial images** and highlighting **pollution-like regions** (e.g., dark oil patches or high-contrast debris) using **OpenCV**. Built with Streamlit for fast iteration and demoability.

## 🔧 Local Setup

```bash
# 1) Create and enter a folder
mkdir mareye-drone-mvp && cd mareye-drone-mvp

# 2) Copy these 3 files into this folder:
#    - drone_app.py
#    - requirements.txt
#    - README.md

# 3) (Optional) Create a virtual environment
python -m venv .venv            # use python3 on macOS/Linux if needed
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 4) Install dependencies
pip install -r requirements.txt

# 5) Run the app
streamlit run drone_app.py
```

Your browser should open at **http://localhost:8501**. Upload an aerial image and tweak the sidebar settings.

## 🚀 Deploy to Streamlit Cloud

1. Push this folder to a **public GitHub repo** (e.g., `mareye-drone-mvp`).
2. Go to **share.streamlit.io**, sign in, and connect your GitHub.
3. Select the repo and set **App file** to `drone_app.py`.
4. Keep defaults and **Deploy**.

> We use `opencv-python-headless` to ensure Streamlit Cloud builds cleanly.

## ✍️ Suggested Submission Text

> The **Drone MVP** is implemented as a Streamlit web app. Since the team doesn’t yet use a physical drone for the MVP, the app simulates the drone by processing uploaded **aerial images** with **OpenCV** to highlight potential pollution-like regions (e.g., dark oil patches or high-contrast debris). **Future work** will integrate **YOLO-based detection** on real drone footage, with onboard edge AI and 4G/5G connectivity.

## 🧭 Notes

- This is a simple heuristic (thresholding, edges, morphology). It’s **not** a production detector.
- Works best on **top-down** aerial photos where pollution appears **darker** than water/background or has **strong edges**.
>>>>>>> 6fa7526 (Initial commit: Mareye Drone MVP)
