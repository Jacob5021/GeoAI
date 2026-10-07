import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

WGS84 = "EPSG:4326"


def read_band_wgs84(src, band=1, max_size=2048):
    """Read one band of an open raster in lat/lon (EPSG:4326), nodata as NaN.

    Downsampled so the longer side is at most max_size pixels (for display).
    Returns (float32 array, BoundingBox in degrees). Rasters without a CRS are
    returned as-is, assuming their bounds are already lat/lon.
    """
    f = max(1, int(np.ceil(max(src.width, src.height) / max_size)))
    h, w = int(np.ceil(src.height / f)), int(np.ceil(src.width / f))
    data = src.read(band, out_shape=(h, w), masked=True, resampling=Resampling.average)
    data = data.astype("float32").filled(np.nan)
    if src.crs is None or src.crs == WGS84:
        return data, src.bounds

    src_transform = src.transform @ Affine.scale(src.width / w, src.height / h)
    transform, width, height = calculate_default_transform(src.crs, WGS84, w, h, *src.bounds)
    dst = np.full((height, width), np.nan, dtype="float32")
    reproject(
        source=data, destination=dst,
        src_transform=src_transform, src_crs=src.crs, src_nodata=np.nan,
        dst_transform=transform, dst_crs=WGS84, dst_nodata=np.nan,
        resampling=Resampling.bilinear)
    west, south, east, north = array_bounds(height, width, transform)
    return dst, BoundingBox(west, south, east, north)


def to_geotiff_bytes(array, source_file, nodata=None):
    """Encode a 2D array as a GeoTIFF using the georeferencing of source_file."""
    source_file.seek(0)
    with rasterio.open(source_file) as src:
        crs, transform = src.crs, src.transform
    source_file.seek(0)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                      dtype=array.dtype, crs=crs, transform=transform, nodata=nodata,
                      compress="deflate") as dst:
            dst.write(array, 1)
        return mem.read()


def validate_pollution_data(df, lat_col, lon_col, pollution_col):
    """Comprehensive pollution data validation"""
    errors = []
    
    # Column checks
    if lat_col not in df.columns:
        errors.append(f"Missing latitude column: {lat_col}")
    if lon_col not in df.columns:
        errors.append(f"Missing longitude column: {lon_col}")
    if pollution_col not in df.columns:
        errors.append(f"Missing pollution column: {pollution_col}")
    
    # Data checks
    if not errors:
        try:
            if not pd.api.types.is_numeric_dtype(df[pollution_col]):
                errors.append("Pollution data must be numeric")
                
            if (df[lat_col] < -90).any() or (df[lat_col] > 90).any():
                errors.append("Latitude out of range (-90 to 90)")
                
            if (df[lon_col] < -180).any() or (df[lon_col] > 180).any():
                errors.append("Longitude out of range (-180 to 180)")
        except:
            errors.append("Data validation failed")
    
    return {
        'is_valid': len(errors) == 0,
        'message': "; ".join(errors) if errors else "Data is valid"
    }