import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import numpy as np

def show_about():
    """Display about information"""
    st.header("About Geospatial AI Tools")
    st.write("""
    This application provides a suite of tools for analyzing satellite and geospatial data.
    
    **Features include:**
    - NDVI calculation and visualization
    - Land use classification
    - GPS heatmap generation
    - Pollution data analysis
    - Crop health monitoring
    - Object detection in satellite imagery
    """)
    
    st.markdown("""
    ### How to Use
    1. Start by uploading your data in the **Data Uploader** tab
    2. Navigate to the tool you want to use
    3. The tool will automatically detect relevant files from your uploads
    4. Adjust parameters as needed and view results
    """)

def plot_ndvi(ndvi_array, title="NDVI Map"):
    """Plot NDVI array"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ndvi_array, cmap='YlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='NDVI Value')
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)

def create_heatmap(data, latitude_col='lat', longitude_col='lon', zoom=12):
    """Create interactive heatmap"""
    m = folium.Map(location=[data[latitude_col].mean(), data[longitude_col].mean()], zoom_start=zoom)
    heat_data = [[row[latitude_col], row[longitude_col]] for _, row in data.iterrows()]
    HeatMap(heat_data).add_to(m)
    folium_static(m)