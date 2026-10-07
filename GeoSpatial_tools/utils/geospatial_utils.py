import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.control import GroundControlPoint
from rasterio.coords import BoundingBox
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds, from_gcps
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
    data = src.read(band, out_shape=(h, w), resampling=Resampling.average).astype("float32")
    data[src.dataset_mask(out_shape=(h, w)) == 0] = np.nan  # nodata, alpha band or internal mask
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


def utm_crs(lon, lat):
    """WGS84 / UTM zone containing (lon, lat)."""
    zone = int((lon + 180) // 6) % 60 + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def fit_gcps(gcps):
    """Least-squares affine from ground control points [(col, row, lon, lat), ...].

    Fitted in the local UTM zone so residuals are in metres.
    Returns (crs, transform, residuals_m).
    """
    from pyproj import Transformer
    if len(gcps) < 3:
        raise ValueError("Add at least 3 control points (4-6 spread across the image is better)")
    cols, rows, lons, lats = (np.array(v, dtype=float) for v in zip(*gcps))
    crs = utm_crs(lons.mean(), lats.mean())
    xs, ys = Transformer.from_crs(WGS84, crs, always_xy=True).transform(lons, lats)
    # Collinear points can't pin down rotation/scale
    c = np.cov(cols, rows)
    if c[0, 0] == 0 or c[1, 1] == 0 or 1 - c[0, 1] ** 2 / (c[0, 0] * c[1, 1]) < 1e-3:
        raise ValueError("Control points are (nearly) in a line; spread them across the image")
    transform = from_gcps([GroundControlPoint(row=r, col=c, x=x, y=y) for c, r, x, y in zip(cols, rows, xs, ys)])
    px = transform.a * cols + transform.b * rows + transform.c
    py = transform.d * cols + transform.e * rows + transform.f
    return crs, transform, np.hypot(px - xs, py - ys)


def warp_preview(rgba, crs, transform, full_width, full_height):
    """Reproject a small RGBA preview of a georeferenced image to EPSG:4326 for web maps.

    Returns (RGBA array, BoundingBox in degrees)."""
    h, w = rgba.shape[:2]
    src_transform = transform @ Affine.scale(full_width / w, full_height / h)
    west, south, east, north = array_bounds(h, w, src_transform)
    dst_transform, dw, dh = calculate_default_transform(crs, WGS84, w, h, west, south, east, north)
    out = np.zeros((4, dh, dw), dtype=np.uint8)
    for i in range(4):
        reproject(source=np.ascontiguousarray(rgba[..., i]), destination=out[i], src_transform=src_transform, src_crs=crs,
                  dst_transform=dst_transform, dst_crs=WGS84, resampling=Resampling.nearest, dst_nodata=0)
    west, south, east, north = array_bounds(dh, dw, dst_transform)
    return np.moveaxis(out, 0, -1), BoundingBox(west, south, east, north)


def georeferenced_geotiff(data, crs, transform):
    """Copy a raster (GeoTIFF bytes) or an RGB(A) uint8 array into a GeoTIFF with the given georeferencing."""
    with MemoryFile() as out:
        if isinstance(data, (bytes, bytearray)):
            with MemoryFile(data) as mem, mem.open() as src:
                profile = src.profile | {"driver": "GTiff", "crs": crs, "transform": transform, "compress": "deflate"}
                profile.pop("blockxsize", None), profile.pop("blockysize", None)
                with out.open(**profile) as dst:
                    dst.write(src.read())
                    dst.colorinterp = src.colorinterp
        else:
            bands = np.moveaxis(data, -1, 0)
            with out.open(driver="GTiff", height=data.shape[0], width=data.shape[1], count=bands.shape[0],
                          dtype="uint8", crs=crs, transform=transform, photometric="RGB", compress="deflate") as dst:
                dst.write(bands)
        return out.read()


LAT_NAMES = ["lat", "latitude", "y", "ycoord", "y_coord"]
LON_NAMES = ["lon", "lng", "long", "longitude", "x", "xcoord", "x_coord"]


def find_column(df, candidates):
    """First column whose name matches a candidate (case-insensitive), else None."""
    lower = {str(c).lower(): c for c in df.columns}
    return next((lower[c] for c in candidates if c in lower), None)


def clean_points(df, lat_col, lon_col, value_col=None):
    """Numeric lat/lon(/value) rows with valid coordinates; raises ValueError on bad input."""
    cols = [c for c in (lat_col, lon_col, value_col) if c]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(map(str, missing))}")
    out = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if out.empty:
        raise ValueError("No numeric rows in the selected columns")
    if not (out[lat_col].between(-90, 90).all() and out[lon_col].between(-180, 180).all()):
        raise ValueError("Coordinates out of range: latitude must be within ±90 and longitude within ±180")
    return out
