import os
import matplotlib.pyplot as plt
import streamlit as st
from data_uploader.app import data_uploader
from ndvi_viewer.app import ndvi_viewer
from landuse_classifier.app import landuse_classifier
from gps_heatmapper.app import gps_heatmapper
from pollution_visualizer.app import pollution_visualizer
from crop_monitoring.app import crop_monitor
from satellite_detection.app import satellite_detector

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
st.set_page_config(page_title="GeoAI Tools", page_icon=os.path.join(ASSETS, "icon.svg"), layout="wide")
st.logo(os.path.join(ASSETS, "logo.svg"), icon_image=os.path.join(ASSETS, "icon.svg"), size="large")

# One chart style for every tool, matching the app theme
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#D5DEE8", "axes.labelcolor": "#14263D", "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#EEF2F7", "xtick.color": "#5B6B7F", "ytick.color": "#5B6B7F",
    "font.family": "sans-serif", "font.size": 10,
    "axes.prop_cycle": plt.cycler(color=["#1B6CA8", "#2BA36B", "#E8A33D", "#C8553D", "#7A5BA6"]),
})

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.hero h1 {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.2rem;
    background: linear-gradient(90deg, #1B6CA8, #2BA36B);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p {font-size: 1.1rem; opacity: 0.75; margin-top: 0;}
[data-testid="stVerticalBlockBorderWrapper"] {transition: box-shadow .15s, transform .15s;}
[data-testid="stVerticalBlockBorderWrapper"]:hover {box-shadow: 0 6px 18px rgba(27,108,168,.15); transform: translateY(-2px);}
</style>
""", unsafe_allow_html=True)

files = st.session_state.setdefault("uploaded_files", {})

# Uploaded files are shared BytesIO objects; previews and other tools leave the
# cursor mid-file, so rewind before every tool run.
for _files in files.values():
    for _f in _files:
        _f.seek(0)


def tool_page(fn, title, icon, slug):
    return st.Page(lambda: fn(st.session_state.uploaded_files), title=title, icon=icon, url_path=slug)


upload = st.Page(data_uploader, title="Data Uploader", icon=":material/upload_file:", url_path="upload")
tools = {
    "Vegetation": [
        (tool_page(ndvi_viewer, "NDVI Viewer", ":material/eco:", "ndvi"),
         "Vegetation index maps from Sentinel-2, Landsat, MODIS or RGB imagery."),
        (tool_page(crop_monitor, "Crop Monitoring", ":material/agriculture:", "crops"),
         "Track NDVI over time and flag crop stress."),
        (tool_page(landuse_classifier, "Land Use Classifier", ":material/landscape:", "landuse"),
         "Classify water, vegetation, built-up and bare land."),
    ],
    "Mapping": [
        (tool_page(gps_heatmapper, "GPS Heatmapper", ":material/local_fire_department:", "heatmap"),
         "Interactive heatmaps from GPS points, uploaded or drawn."),
        (tool_page(pollution_visualizer, "Pollution Visualizer", ":material/air:", "pollution"),
         "NO₂, PM2.5 and other pollutants from stations or rasters."),
    ],
    "AI Detection": [
        (tool_page(satellite_detector, "Object Detection", ":material/satellite_alt:", "detect"),
         "YOLOv8 object detection on imagery."),
    ],
}


def home():
    st.markdown("<div class='hero'><h1>GeoAI Tools</h1>"
                "<p>Upload satellite rasters, imagery or GPS tables once, then analyse them with any tool.</p></div>",
                unsafe_allow_html=True)

    n_files = sum(len(v) for v in files.values())
    c1, c2, _ = st.columns(3)
    c1.metric("Files loaded", n_files)
    c2.metric("Tools", sum(len(v) for v in tools.values()))
    if not n_files:
        st.page_link(upload, label="Start by uploading data", icon=":material/arrow_forward:")

    for section, pages in tools.items():
        st.subheader(section)
        cols = st.columns(3)
        for i, (page, blurb) in enumerate(pages):
            with cols[i % 3].container(border=True):
                st.page_link(page, label=f"**{page.title}**", icon=page.icon)
                st.caption(blurb)


nav = st.navigation({
    "": [st.Page(home, title="Home", icon=":material/home:", default=True), upload],
    **{section: [p for p, _ in pages] for section, pages in tools.items()},
})

with st.sidebar:
    st.caption(f":material/folder: {sum(len(v) for v in files.values())} file(s) loaded")

nav.run()
