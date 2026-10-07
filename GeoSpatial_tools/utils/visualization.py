import streamlit as st
import folium
import xyzservices.providers as xyz
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import numpy as np

def page_header(title, subtitle):
    """Consistent title + one-line description at the top of every tool."""
    st.title(title)
    st.caption(subtitle)
    st.divider()


def empty_state(message):
    """Friendly prompt shown when a tool has no matching uploads."""
    st.info(f"{message} Add files in **Data Uploader** (sidebar).", icon=":material/upload_file:")


def plot_ndvi(ndvi_array, title="NDVI Map"):
    """Plot NDVI array"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ndvi_array, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='NDVI Value')
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

BASEMAPS = {  # all key-free (CartoDB tiles now require an API key)
    "Light Gray": xyz.Esri.WorldGrayCanvas,
    "OpenStreetMap": xyz.OpenStreetMap.Mapnik,
    "Satellite": xyz.Esri.WorldImagery,
}


def add_basemaps(m):
    """Add switchable basemaps (first is the default) and a layer control to a folium map."""
    for name, provider in BASEMAPS.items():
        folium.TileLayer(provider, name=name).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def display_map(folium_map, width=None, height=500):
    """Render a folium map full-width (or at a fixed width) without round-tripping map events."""
    st_folium(folium_map, width=width, height=height, use_container_width=width is None, returned_objects=[])


def prepare_for_display(img_array):
    """Scale any numeric image to uint8 [0-255] for display/PIL/OpenCV (NaN -> 0)."""
    src = np.asarray(img_array)
    arr = np.nan_to_num(src.astype(float))
    if np.issubdtype(src.dtype, np.floating) and arr.min() >= 0 and arr.max() <= 1:
        arr = arr * 255
    elif np.issubdtype(src.dtype, np.floating) or arr.min() < 0 or arr.max() > 255:
        arr = (arr - arr.min()) / (np.ptp(arr) + 1e-8) * 255
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)
