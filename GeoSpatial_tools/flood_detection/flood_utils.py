# flood_utils.py
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

# ================== Raster IO ==================
def read_raster_as_array(path, rescale_to=None):
    """
    Read single-band raster and return (arr, profile).
    If rescale_to is provided (path of reference raster), resample input
    to match its extent/resolution/CRS.
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile

        if rescale_to is not None:
            with rasterio.open(rescale_to) as ref:
                dst_arr = np.empty((ref.height, ref.width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=Resampling.bilinear
                )
                arr = dst_arr
                profile.update({
                    "height": ref.height,
                    "width": ref.width,
                    "transform": ref.transform,
                    "crs": ref.crs
                })

    return arr, profile

def save_array_as_geotiff(path, arr, ref_profile, dtype=rasterio.float32):
    """
    Save a NumPy array as GeoTIFF using reference profile (from input DEM/precip).
    """
    profile = ref_profile.copy()
    profile.update(dtype=dtype, count=1)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)

def save_array_as_png(path, arr, cmap="viridis", vmin=None, vmax=None):
    """
    Save array as a PNG (normalized 0–255, colored).
    """
    import matplotlib.pyplot as plt
    plt.imsave(path, arr, cmap=cmap, vmin=vmin, vmax=vmax)

# ================== Terrain Analysis ==================
def compute_slope(elevation_array, cellsize=None):
    """
    Compute slope from elevation array.
    slope = sqrt((dz/dx)^2 + (dz/dy)^2)
    """
    dzdx = np.gradient(elevation_array, axis=1)
    dzdy = np.gradient(elevation_array, axis=0)
    if cellsize:
        dzdx = dzdx / cellsize
        dzdy = dzdy / cellsize
    slope = np.sqrt(dzdx**2 + dzdy**2)
    return np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)

# ================== Normalization ==================
def normalize(arr, clip_quantiles=(0.02, 0.98)):
    """Normalize to 0–1 with quantile clipping."""
    if arr.size == 0:
        return arr
    low, high = np.nanpercentile(arr, [clip_quantiles[0]*100, clip_quantiles[1]*100])
    arr_clipped = np.clip(arr, low, high)
    denom = (high - low) if (high - low) != 0 else 1.0
    norm = (arr_clipped - low) / denom
    return np.nan_to_num(norm, nan=0.0)

# ================== Flood Risk ==================
def compute_flood_risk_index(elev_arr, precip_arr, slope_arr=None, weights=None):
    """
    Compute Flood Risk Index (FRI) from:
      - precipitation (higher = higher risk)
      - elevation (lower = higher risk)
      - slope (flatter = higher risk)
    Returns normalized 0–1 FRI.
    """
    elev_norm = normalize(elev_arr)
    precip_norm = normalize(precip_arr)

    if slope_arr is None:
        slope_arr = compute_slope(elev_arr)
    slope_norm = normalize(slope_arr)

    elev_inv = 1.0 - elev_norm   # low elevation = high risk
    slope_inv = 1.0 - slope_norm # flat = high risk

    if weights is None:
        weights = {"precip": 0.5, "elev": 0.3, "slope": 0.2}

    fri = (weights["precip"] * precip_norm +
           weights["elev"] * elev_inv +
           weights["slope"] * slope_inv)

    fri_smooth = gaussian_filter(fri, sigma=1)
    return normalize(fri_smooth, clip_quantiles=(0.0, 1.0))

def classify_fri(fri_arr, thresholds=(0.3, 0.6)):
    """
    Classify flood risk into 3 levels:
      0 = Low, 1 = Medium, 2 = High
    """
    classes = np.zeros_like(fri_arr, dtype=np.uint8)
    classes[fri_arr < thresholds[0]] = 0
    classes[(fri_arr >= thresholds[0]) & (fri_arr < thresholds[1])] = 1
    classes[fri_arr >= thresholds[1]] = 2
    return classes
