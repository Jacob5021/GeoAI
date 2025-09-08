import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import rasterio
from PIL import Image

from .ndvi_processor import (
    calculate_ndvi,
    get_satellite_bands,
    SATELLITE_PROFILES
)
from utils.visualization import plot_ndvi


# --- Helper: load bands with no-data mask ---
def load_bands_with_mask(file, red_idx, nir_idx):
    with rasterio.open(file) as src:
        red = src.read(red_idx).astype(float)
        nir = src.read(nir_idx).astype(float)

        # Detect no-data value
        nodata = src.nodata
        if nodata is not None:
            mask = (red == nodata) | (nir == nodata)
        else:
            # Fallback: treat pure black pixels as no-data
            mask = (red == 0) & (nir == 0)

        # Apply mask
        red[mask] = np.nan
        nir[mask] = np.nan

    return red, nir


# --- Main NDVI Viewer ---
def ndvi_viewer(uploaded_files):
    st.header("🌱 Advanced NDVI Viewer")
    st.markdown("""
    Calculate vegetation index using:
    - Predefined satellite band profiles (GeoTIFF)
    - **Image mode** (RGB with proxy NIR = Green channel)
    """)

    # --- File selection ---
    image_files = []
    for ext in ['tif', 'tiff', 'geotiff', 'jpg', 'jpeg', 'png']:
        image_files.extend(uploaded_files.get(ext, []))

    if not image_files:
        st.warning("Please upload satellite images in Data Uploader first")
        return

    selected_file = st.selectbox("Select image", [f.name for f in image_files])
    file = next(f for f in image_files if f.name == selected_file)

    # --- Detect file type (GeoTIFF vs RGB) ---
    try:
        with rasterio.open(file) as src:
            total_bands = src.count
            is_geotiff = True
    except Exception:
        img = np.array(Image.open(file))
        total_bands = img.shape[-1] if img.ndim == 3 else 1
        is_geotiff = False

    # --- Satellite/Sensor selection ---
    sat_options = list(SATELLITE_PROFILES.keys()) + ["Image (RGB with proxy NIR)"]

    if is_geotiff:
        default_index = 0
    else:
        default_index = len(sat_options) - 1

    satellite = st.selectbox(
        "Satellite/Sensor",
        options=sat_options,
        index=default_index,
        help="Choose a satellite profile or 'Image' for RGB photos"
    )

    # --- Band selection logic ---
    if satellite == "Image (RGB with proxy NIR)":
        red_idx, nir_idx = 0, 1  # RGB → R = 0, G = 1
        st.warning("⚠️ Using RGB image mode: Red = R, NIR = Green (proxy).")
    else:
        band_info = get_satellite_bands(satellite)
        col1, col2 = st.columns(2)
        with col1:
            if satellite == 'Custom':
                red_idx = st.number_input("Red band index", min_value=1, value=3, step=1)
            else:
                red_idx = band_info['red']
                st.info(f"Using Red Band {red_idx} for {satellite}")
        with col2:
            if satellite == 'Custom':
                nir_idx = st.number_input("NIR band index", min_value=1, value=4, step=1)
            else:
                nir_idx = band_info['nir']
                st.info(f"Using NIR Band {nir_idx} for {satellite}")

        with st.expander("📊 View Band Information"):
            if satellite != 'Custom' and band_info.get('bands'):
                st.markdown(f"**{satellite} Bands:**")
                for band, desc in band_info['bands'].items():
                    st.markdown(f"- Band {band}: {desc}")
            else:
                st.info("No band information available for custom selection")

    # --- Processing ---
    if st.button("🌿 Calculate NDVI", type="primary"):
        try:
            with st.spinner("Processing..."):
                if satellite == "Image (RGB with proxy NIR)":
                    img = np.array(Image.open(file))
                    red = img[:, :, red_idx].astype(float)
                    nir = img[:, :, nir_idx].astype(float)
                    ndvi = calculate_ndvi(red, nir)
                    mode = "Proxy NDVI"
                else:
                    if is_geotiff and total_bands >= max(red_idx, nir_idx):
                        red, nir = load_bands_with_mask(file, red_idx, nir_idx)
                        ndvi = calculate_ndvi(red, nir)
                        # Mask NDVI as well
                        ndvi = np.where(np.isnan(red) | np.isnan(nir), np.nan, ndvi)
                        mode = "Real NDVI"
                    else:
                        st.error(f"Selected bands not available. Image has {total_bands} bands.")
                        return

                # --- Visualization ---
                tab1, tab2 = st.tabs(["NDVI Map", "Band Preview"])

                with tab1:
                    plot_ndvi(ndvi, f"{satellite} {mode} - {file.name}")

                    # Download
                    output = BytesIO()
                    plt.imsave(output, ndvi, format='png', cmap='RdYlGn', vmin=-1, vmax=1)
                    output.seek(0)
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
                        st.metric("Red Band", f"Band {red_idx}" if mode == "Real NDVI" else "RGB: R")
                        st.image(red, caption="Red Band", use_container_width=True)
                    with col2:
                        st.metric("NIR Band", f"Band {nir_idx}" if mode == "Real NDVI" else "RGB: G (proxy)")
                        st.image(nir, caption="NIR Band", use_container_width=True)

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            st.info("ℹ️ Tips: Check if band indices match your image's actual bands")
