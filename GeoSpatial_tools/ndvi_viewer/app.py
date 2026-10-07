import streamlit as st
from utils.visualization import page_header, empty_state
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import rasterio
from PIL import Image

from .ndvi_processor import (
    calculate_ndvi,
    load_bands_with_mask,
    get_satellite_bands,
    SATELLITE_PROFILES
)
from utils.visualization import plot_ndvi, prepare_for_display
from utils.geospatial_utils import to_geotiff_bytes


# --- Main NDVI Viewer ---
def ndvi_viewer(uploaded_files):
    page_header("NDVI Viewer", "Vegetation index from multispectral GeoTIFFs (Sentinel-2, Landsat, MODIS) or RGB proxy.")

    # --- File selection ---
    image_files = []
    for ext in ['tif', 'tiff', 'geotiff', 'jpg', 'jpeg', 'png']:
        image_files.extend(uploaded_files.get(ext, []))

    if not image_files:
        empty_state("No imagery yet.")
        return

    selected_file = st.selectbox("Select image", [f.name for f in image_files])
    file = next(f for f in image_files if f.name == selected_file)

    # --- Detect file type (GeoTIFF vs RGB) ---
    # GDAL opens PNG/JPG too, so decide by extension
    is_geotiff = file.name.lower().endswith(('.tif', '.tiff', '.geotiff'))
    if is_geotiff:
        with rasterio.open(file) as src:
            total_bands = src.count
    else:
        img = np.array(Image.open(file))
        total_bands = img.shape[-1] if img.ndim == 3 else 1

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
        key=f"ndvi_sat_{selected_file}",  # re-default when the file changes
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
                        ndvi = calculate_ndvi(red, nir)  # NaN bands stay NaN
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
                    if mode == "Real NDVI":
                        st.download_button(
                            "🗺️ Download NDVI GeoTIFF",
                            to_geotiff_bytes(ndvi.astype("float32"), file, nodata=np.nan),
                            file_name=f"ndvi_{satellite}_{file.name.split('.')[0]}.tif",
                            mime="image/tiff"
                        )

                with tab2:
                    st.subheader("Input Bands")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Red Band", f"Band {red_idx}" if mode == "Real NDVI" else "RGB: R")
                        st.image(prepare_for_display(red), caption="Red Band", width="stretch")
                    with col2:
                        st.metric("NIR Band", f"Band {nir_idx}" if mode == "Real NDVI" else "RGB: G (proxy)")
                        st.image(prepare_for_display(nir), caption="NIR Band", width="stretch")

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            st.info("ℹ️ Tips: Check if band indices match your image's actual bands")
