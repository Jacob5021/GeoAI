import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from .ndvi_processor import (
    calculate_ndvi,
    load_bands,
    get_satellite_bands,
    SATELLITE_PROFILES
)
from utils.visualization import plot_ndvi

def ndvi_viewer(uploaded_files):
    st.header("🌱 Advanced NDVI Viewer")
    st.markdown("""
    Calculate vegetation index using:
    - Predefined satellite band profiles **OR**
    - Custom band indices
    """)
    
    # File selection
    image_files = []
    for ext in ['tif', 'tiff', 'geotiff', 'jpg', 'png']:
        image_files.extend(uploaded_files.get(ext, []))
    
    if not image_files:
        st.warning("Please upload satellite images in Data Uploader first")
        return
    
    selected_file = st.selectbox("Select image", [f.name for f in image_files])
    file = next(f for f in image_files if f.name == selected_file)
    
    # Satellite selection
    satellite = st.selectbox(
        "Satellite/Sensor",
        options=list(SATELLITE_PROFILES.keys()),
        index=0,
        help="Select 'Custom' to manually specify bands"
    )
    
    band_info = get_satellite_bands(satellite)
    
    # Band configuration
    col1, col2 = st.columns(2)
    
    with col1:
        if satellite == 'Custom':
            red_idx = st.number_input(
                "Red band index",
                min_value=1,
                value=3,
                step=1,
                help="1-based index (Band 3 for RGB, Band 4 for Sentinel-2)"
            )
        else:
            red_idx = st.number_input(
                f"Red band (default: {band_info['red']})",
                min_value=1,
                value=band_info['red'],
                step=1,
                help=f"Default: Band {band_info['red']} for {satellite}"
            )
    
    with col2:
        if satellite == 'Custom':
            nir_idx = st.number_input(
                "NIR band index",
                min_value=1,
                value=4,
                step=1,
                help="1-based index (Band 4 for RGB, Band 8 for Sentinel-2)"
            )
        else:
            nir_idx = st.number_input(
                f"NIR band (default: {band_info['nir']})",
                min_value=1,
                value=band_info['nir'],
                step=1,
                help=f"Default: Band {band_info['nir']} for {satellite}"
            )
    
    # Band information expander
    with st.expander("📊 View Band Information"):
        if satellite != 'Custom' and band_info.get('bands'):
            st.markdown(f"**{satellite} Bands:**")
            for band, desc in band_info['bands'].items():
                st.markdown(f"- Band {band}: {desc}")
        else:
            st.info("No band information available for custom selection")
    
    if st.button("🌿 Calculate NDVI", type="primary"):
        try:
            with st.spinner("Processing..."):
                red, nir = load_bands(file, red_idx, nir_idx)
                ndvi = calculate_ndvi(red, nir)
                
                # Visualization
                tab1, tab2 = st.tabs(["NDVI Map", "Band Preview"])
                
                with tab1:
                    plot_ndvi(ndvi, f"{satellite} NDVI - {file.name}")
                    
                    # Download
                    output = BytesIO()
                    plt.imsave(output, ndvi, format='png', cmap='YlGn', vmin=-1, vmax=1)
                    st.download_button(
                        "💾 Download NDVI Map",
                        output.getvalue(),
                        file_name=f"ndvi_{satellite}_{file.name.split('.')[0]}.png",
                        mime="image/png"
                    )
                
                with tab2:
                    st.subheader("Input Bands")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Red Band", f"Band {red_idx}")
                        st.image(red, caption=f"Red Band {red_idx}", use_column_width=True)
                    with col2:
                        st.metric("NIR Band", f"Band {nir_idx}")
                        st.image(nir, caption=f"NIR Band {nir_idx}", use_column_width=True)
        
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            st.info("ℹ️ Tips: Check if band indices match your image's actual bands")