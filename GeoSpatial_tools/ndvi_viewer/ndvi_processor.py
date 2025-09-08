import rasterio
import numpy as np
from PIL import Image
import tempfile
import os

# Satellite band configurations
SATELLITE_PROFILES = {
    'Sentinel-2': {
        'description': "ESA Copernicus Sentinel-2",
        'red': 4,  # Band 4 (Red)
        'nir': 8,  # Band 8 (NIR)
        'bands': {
            1: 'Coastal aerosol (443nm)',
            2: 'Blue (490nm)',
            3: 'Green (560nm)',
            4: 'Red (665nm)',
            5: 'Vegetation Red Edge (705nm)',
            6: 'Vegetation Red Edge (740nm)',
            7: 'Vegetation Red Edge (783nm)',
            8: 'NIR (842nm)',
            9: 'Water vapour (945nm)',
            10: 'SWIR - Cirrus (1375nm)',
            11: 'SWIR (1610nm)',
            12: 'SWIR (2190nm)'
        }
    },
    'Landsat-8': {
        'description': "USGS Landsat 8",
        'red': 4,  # Band 4 (Red)
        'nir': 5,  # Band 5 (NIR)
        'bands': {
            1: 'Coastal (433-453nm)',
            2: 'Blue (450-515nm)',
            3: 'Green (525-600nm)',
            4: 'Red (630-680nm)',
            5: 'NIR (845-885nm)',
            6: 'SWIR 1 (1560-1660nm)',
            7: 'SWIR 2 (2100-2300nm)',
            8: 'Panchromatic (500-680nm)',
            9: 'Cirrus (1360-1380nm)'
        }
    },
    'MODIS': {
        'description': "NASA MODIS",
        'red': 1,
        'nir': 2,
        'bands': {
            1: 'Red (620-670nm)',
            2: 'NIR (841-876nm)',
            3: 'Blue-Green (459-479nm)',
            4: 'Green (545-565nm)',
            5: 'NIR (1230-1250nm)',
            6: 'SWIR (1628-1652nm)',
            7: 'SWIR (2105-2155nm)'
        }
    },
    'Custom': {
        'description': "User-defined bands",
        'red': None,
        'nir': None,
        'bands': {}
    }
}

def get_satellite_bands(satellite):
    """Get band information for selected satellite"""
    return SATELLITE_PROFILES.get(satellite, SATELLITE_PROFILES['Custom'])

def load_bands(file_input, red_idx, nir_idx):
    """
    Load bands from image file with validation - handles both file paths and UploadedFile objects

    Args:
        file_input: File path string or Streamlit UploadedFile object
        red_idx: 1-based index for red band
        nir_idx: 1-based index for NIR band

    Returns:
        tuple: (red_band, nir_band)
    """
    # Check if input is a Streamlit UploadedFile object
    if hasattr(file_input, 'read') and hasattr(file_input, 'name'):
        # Handle UploadedFile object
        file_name = file_input.name.lower()

        if file_name.endswith(('.tif', '.tiff', '.geotiff')):
            # Create temporary file for rasterio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                tmp_file.write(file_input.getvalue())
                tmp_path = tmp_file.name

            try:
                with rasterio.open(tmp_path) as src:
                    if red_idx > src.count or nir_idx > src.count:
                        raise ValueError(f"Band indices must be between 1 and {src.count}")
                    red_band = src.read(red_idx)
                    nir_band = src.read(nir_idx)
            finally:
                os.unlink(tmp_path)  # Clean up temp file

            return red_band, nir_band

        else:  # For RGB/PNG/JPG
            img = Image.open(file_input)
            arr = np.array(img)
            if len(arr.shape) != 3:
                raise ValueError("Grayscale images not supported for NDVI")
            if red_idx-1 >= arr.shape[2] or nir_idx-1 >= arr.shape[2]:
                raise ValueError(f"Channel indices must be between 1 and {arr.shape[2]}")
            return arr[:, :, red_idx-1], arr[:, :, nir_idx-1]

    else:
        # Handle file path string (original logic)
        if file_input.lower().endswith(('.tif', '.tiff', '.geotiff')):
            with rasterio.open(file_input) as src:
                if red_idx > src.count or nir_idx > src.count:
                    raise ValueError(f"Band indices must be between 1 and {src.count}")
                return src.read(red_idx), src.read(nir_idx)
        else:  # For RGB/PNG/JPG
            img = Image.open(file_input)
            arr = np.array(img)
            if len(arr.shape) != 3:
                raise ValueError("Grayscale images not supported for NDVI")
            if red_idx-1 >= arr.shape[2] or nir_idx-1 >= arr.shape[2]:
                raise ValueError(f"Channel indices must be between 1 and {arr.shape[2]}")
            return arr[:, :, red_idx-1], arr[:, :, nir_idx-1]

def calculate_ndvi(red_band, nir_band):
    """Calculate NDVI with safety checks"""
    red = red_band.astype(float)
    nir = nir_band.astype(float)

    # Mask division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(
            (nir + red) != 0,
            (nir - red) / (nir + red),
            0
        )

    return np.clip(ndvi, -1, 1)

def calculate_ndvi_from_file(file_input, red_idx: int, nir_idx: int) -> np.ndarray:
    """
    Convenience function to calculate NDVI directly from file

    Args:
        file_input: File path string or UploadedFile object
        red_idx: 1-based red band index
        nir_idx: 1-based NIR band index

    Returns:
        2D NDVI array with same dimensions as input bands
    """
    red, nir = load_bands(file_input, red_idx, nir_idx)
    return calculate_ndvi(red, nir)

def get_mean_ndvi(file_input, red_idx: int, nir_idx: int) -> float:
    """
    Calculate mean NDVI for crop monitoring

    Args:
        file_input: File path string or UploadedFile object
        red_idx: 1-based red band index
        nir_idx: 1-based NIR band index

    Returns:
        Mean NDVI value (NaN if calculation fails)
    """
    try:
        ndvi = calculate_ndvi_from_file(file_input, red_idx, nir_idx)
        return np.nanmean(ndvi)  # Ignores NaN values
    except Exception:
        return np.nan
