import streamlit as st
from data_uploader.app import data_uploader
from ndvi_viewer.app import ndvi_viewer
from landuse_classifier.app import landuse_classifier
from gps_heatmapper.app import gps_heatmapper
from pollution_visualizer.app import pollution_visualizer
from crop_monitoring.app import crop_monitor
from satellite_detection.app import satellite_detector
from flood_detection.app import flood_detector, flood_predictor
from utils.visualization import show_about

# ============= PAGE CONFIGURATION & THEME =============
st.set_page_config(
    page_title="Geospatial AI Tools",
    page_icon="🌍",
    layout="wide",
    menu_items={
        "About": "## Powerful Geospatial AI tools for satellite analytics and more!"
    }
)

# ============= CUSTOM CSS FOR AESTHETICS =============
st.markdown("""
    <style>
    .sidebar .sidebar-content {background-color: #f0f3fa;}
    .main-title {
        font-size: 2.75rem !important;
        font-weight: 700;
        letter-spacing: -1px;
        color: #13386a;
        margin-bottom: 0.8em;
    }
    .tool-header {
        font-size: 1.3rem !important;
        font-weight: 500;
        color: #1B6CA8;
        border-left: 4px solid #1B6CA8;
        padding-left: 0.3em;
        margin-bottom: 0.5em;
    }
    .stVerticalBlock {
        margin-top: 1.5em !important;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg,#1b6ca8 60%, #38b9ff 100%);
        border: none;
        font-size: 1.06rem;
        font-weight: 600;
    }
    .stDownloadButton > button {
        background: #1B6CA8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ============= SIDEBAR WITH ICONS & SPACING =============
st.sidebar.markdown(
    """
    <h2 style='margin-bottom:0.2em;'>🌐 Geospatial AI Tools</h2>
    <hr style="margin-top:0.2em; margin-bottom:1em; border:0; height:2px; background-color:#1b6ca8;">
    """, unsafe_allow_html=True
)

tool = st.sidebar.radio(
    "Navigate:",
    [
        "⬆️ Data Uploader",
        "🌱 NDVI Viewer",
        "🏞️ Land Use Classifier",
        "🔥 GPS Heatmapper",
        "🌫️ Pollution Visualizer",
        "🌾 Crop Monitoring",
        "🛰️ Object Detection",
        "🌊 Flood Analysis",
        "ℹ️ About"
    ],
    index=0,
)

# ============= MAIN HEADER AREA =============
st.markdown("<div class='main-title'>🌍 Geospatial AI Tools Suite</div>", unsafe_allow_html=True)

st.markdown("""
<div style='color:#2670b7; font-size:1.15rem; margin-bottom:1.2em;'>
Insights from your earth observation and geospatial data: NDVI analytics, object detection, pollution, heatmaps, monitoring, and more.
</div>
""", unsafe_allow_html=True)

# ============= TOOL SECTION AREA =============
if tool.startswith("⬆️"):
    st.markdown("<div class='tool-header'>Upload Geospatial Data</div>", unsafe_allow_html=True)
    st.session_state.uploaded_files = data_uploader()
elif tool.startswith("🌱"):
    st.markdown("<div class='tool-header'>NDVI Viewer</div>", unsafe_allow_html=True)
    ndvi_viewer(st.session_state.get("uploaded_files", {}))
elif tool.startswith("🏞️"):
    st.markdown("<div class='tool-header'>Land Use Classifier</div>", unsafe_allow_html=True)
    landuse_classifier(st.session_state.get("uploaded_files", {}))
elif tool.startswith("🔥"):
    st.markdown("<div class='tool-header'>GPS Heatmapper</div>", unsafe_allow_html=True)
    gps_heatmapper(st.session_state.get("uploaded_files", {}))
elif tool.startswith("🌫️"):
    st.markdown("<div class='tool-header'>Pollution Visualizer</div>", unsafe_allow_html=True)
    pollution_visualizer(st.session_state.get("uploaded_files", {}))
elif tool.startswith("🌾"):
    st.markdown("<div class='tool-header'>Crop Monitoring</div>", unsafe_allow_html=True)
    crop_monitor(st.session_state.get("uploaded_files", {}))
elif tool.startswith("🛰️"):
    st.markdown("<div class='tool-header'>Object Detection</div>", unsafe_allow_html=True)
    satellite_detector(st.session_state.get("uploaded_files", {}))
elif tool.startswith("🌊"):
    st.markdown("<div class='tool-header'>Flood Analysis</div>", unsafe_allow_html=True)

    uploaded_files = st.session_state.get("uploaded_files", {})

    tab1, tab2 = st.tabs(["🌊 Flood Detection", "📈 Flood Prediction"])
    with tab1:
        flood_detector(uploaded_files)
    with tab2:
        flood_predictor(uploaded_files)
elif tool.startswith("ℹ️"):
    st.markdown("<div class='tool-header'>About This Project</div>", unsafe_allow_html=True)
    show_about()

