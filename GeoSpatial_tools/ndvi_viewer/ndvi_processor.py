import rasterio
import numpy as np

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

def load_bands_with_mask(file, red_idx, nir_idx):
    if hasattr(file, "seek"):
        file.seek(0)
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

def calculate_ndvi_from_file(file, red_idx, nir_idx):
    """NDVI from a raster, nodata pixels as NaN."""
    return calculate_ndvi(*load_bands_with_mask(file, red_idx, nir_idx))
