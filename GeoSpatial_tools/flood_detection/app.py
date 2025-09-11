import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import rasterio

# ================== Satellite Profiles for NDWI ==================
SATELLITE_PROFILES = {
    'Sentinel-2': {'green': 3, 'nir': 8},
    'Landsat-8': {'green': 3, 'nir': 5},
    'MODIS': {'green': 4, 'nir': 2},
    'Custom': {'green': None, 'nir': None},
}

# ================== NDWI Functions ==================
def calculate_ndwi(green_band, nir_band):
    green = green_band.astype(float)
    nir = nir_band.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - nir) / (green + nir + 1e-10)
    return np.clip(ndwi, -1, 1)

def load_bands_with_mask(file, green_idx, nir_idx):
    with rasterio.open(file) as src:
        green = src.read(green_idx).astype(float)
        nir = src.read(nir_idx).astype(float)
        nodata = src.nodata
        if nodata is not None:
            mask = (green == nodata) | (nir == nodata)
        else:
            mask = (green == 0) & (nir == 0)
        green[mask] = np.nan
        nir[mask] = np.nan
    return green, nir

# ================== Flood Detection (NDWI) ==================
def flood_detector(uploaded_files):
    st.subheader("🌊 Flood Detection (NDWI)")
    image_files = []
    for ext in ['tif', 'tiff', 'geotiff', 'jpg', 'jpeg', 'png']:
        image_files.extend(uploaded_files.get(ext, []))
    if not image_files:
        st.warning("Please upload at least one satellite image in Data Uploader.")
        return

    selected_file = st.selectbox("Select image", [f.name for f in image_files])
    file = next(f for f in image_files if f.name == selected_file)

    try:
        with rasterio.open(file) as src:
            total_bands = src.count
            is_geotiff = True
    except Exception:
        img = np.array(Image.open(file))
        total_bands = img.shape[-1] if img.ndim == 3 else 1
        is_geotiff = False

    sat_options = list(SATELLITE_PROFILES.keys()) + ["Image (RGB proxy)"]
    satellite = st.selectbox("Satellite/Sensor", options=sat_options)

    # Band selection
    if satellite == "Image (RGB proxy)":
        green_idx, nir_idx = 1, 2
    else:
        band_info = SATELLITE_PROFILES.get(satellite, SATELLITE_PROFILES['Custom'])
        col1, col2 = st.columns(2)
        with col1:
            if satellite == 'Custom':
                green_idx = st.number_input("Green band index", min_value=1, value=3)
            else:
                green_idx = band_info['green']
                st.info(f"Using Green Band {green_idx}")
        with col2:
            if satellite == 'Custom':
                nir_idx = st.number_input("NIR band index", min_value=1, value=4)
            else:
                nir_idx = band_info['nir']
                st.info(f"Using NIR Band {nir_idx}")

    # Detection
    if st.button("Detect Flooded Areas", type="primary"):
        if satellite == "Image (RGB proxy)":
            img = np.array(Image.open(file))
            green = img[:, :, green_idx].astype(float)
            nir = img[:, :, nir_idx].astype(float)
            ndwi = calculate_ndwi(green, nir)
        else:
            if is_geotiff and total_bands >= max(green_idx, nir_idx):
                green, nir = load_bands_with_mask(file, green_idx, nir_idx)
                ndwi = calculate_ndwi(green, nir)
            else:
                st.error("Selected bands not available.")
                return

        # Plot NDWI
        st.subheader("NDWI Map")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(ndwi, cmap="Blues", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

        # Flood mask
        thresh = st.slider("NDWI Threshold", -1.0, 1.0, 0.2, 0.01)
        flood_mask = (ndwi > thresh).astype(np.uint8)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.imshow(flood_mask, cmap="Blues")
        ax2.set_title("Flood Mask")
        st.pyplot(fig2)

        flooded_pixels = np.nansum(flood_mask)
        total_pixels = np.count_nonzero(~np.isnan(ndwi))
        percent_flooded = flooded_pixels / total_pixels * 100 if total_pixels > 0 else 0
        st.write(f"Flooded area: {percent_flooded:.2f}%")

# ================== Flood Prediction (DEM + Precip) ==================
def flood_predictor(uploaded_files):
    st.subheader("📈 Flood Prediction (DEM + Rainfall)")
    if "dem" not in uploaded_files or "precip" not in uploaded_files:
        st.info("Upload DEM and Precipitation rasters to enable prediction.")
        return

    dem_file = uploaded_files["dem"]
    precip_file = uploaded_files["precip"]

    with rasterio.open(dem_file) as dem_src, rasterio.open(precip_file) as rain_src:
        dem = dem_src.read(1).astype(np.float32)
        precip = resample_to_match(rain_src, dem_src)

    fri = compute_flood_risk_index(dem, precip)
    classes = classify_fri(fri)

    # Show FRI
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(fri, cmap="Reds", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title("Flood Risk Index (0-1)")
    st.pyplot(fig)

    # Show Risk Classes
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap("RdYlBu", 3)
    im2 = ax2.imshow(classes, cmap=cmap, vmin=0, vmax=2)
    cbar = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Low", "Medium", "High"])
    ax2.set_title("Flood Risk Classes")
    st.pyplot(fig2)
