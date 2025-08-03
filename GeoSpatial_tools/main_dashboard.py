import streamlit as st
from data_uploader.app import data_uploader
from ndvi_viewer.app import ndvi_viewer
from landuse_classifier.app import landuse_classifier
from gps_heatmapper.app import gps_heatmapper
from pollution_visualizer.app import pollution_visualizer
from crop_monitoring.app import crop_monitor
from satellite_detection.app import satellite_detector
from utils.visualization import show_about

# Configure page
st.set_page_config(
    page_title="Geospatial AI Tools",
    page_icon="🌍",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("🌐 Geospatial AI Tools")

tool = st.sidebar.radio(
    "Navigation",
    [
        "Data Uploader",
        "NDVI Viewer",
        "Land Use Classifier",
        "GPS Heatmapper",
        "Pollution Visualizer",
        "Crop Monitoring",
        "Satellite Object Detection",
        "About"
    ]
)

# Store session state for uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# Main content area
st.title("Geospatial AI Tools Suite")

if tool == "Data Uploader":
    st.session_state.uploaded_files = data_uploader()
elif tool == "NDVI Viewer":
    ndvi_viewer(st.session_state.get("uploaded_files", {}))
elif tool == "Land Use Classifier":
    landuse_classifier(st.session_state.get("uploaded_files", {}))
elif tool == "GPS Heatmapper":
    gps_heatmapper(st.session_state.get("uploaded_files", {}))
elif tool == "Pollution Visualizer":
    pollution_visualizer(st.session_state.get("uploaded_files", {}))
elif tool == "Crop Monitoring":
    crop_monitor(st.session_state.get("uploaded_files", {}))
elif tool == "Satellite Object Detection":
    satellite_detector(st.session_state.get("uploaded_files", {}))
elif tool == "About":
    show_about()