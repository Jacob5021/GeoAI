"""GeoAI Tools: FastAPI backend + static web frontend.

Run:  uvicorn server:app --port 8599
"""
import io
import os
import re
import tempfile
import uuid
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from rasterio.enums import ColorInterp, Resampling
from rasterio.io import MemoryFile

from landuse_classifier.classifier import available_methods, classify, create_colored_classification_map
from ndvi_viewer.ndvi_processor import SATELLITE_PROFILES, calculate_ndvi, get_satellite_bands, load_bands_with_mask
from utils.geospatial_utils import (LAT_NAMES, LON_NAMES, clean_points, find_column, fit_gcps,
                                    georeferenced_geotiff, read_band_wgs84, to_geotiff_bytes, warp_preview)
from utils.visualization import colorize, png_bytes, png_data_url, prepare_for_display

HERE = os.path.dirname(os.path.abspath(__file__))
RGB_MODE = "Image (RGB with proxy NIR)"
KINDS = {"raster": {"tif", "tiff", "geotiff"}, "image": {"jpg", "jpeg", "png"},
         "table": {"csv"}, "vector": {"geojson", "json", "kml", "gpkg", "zip"}}
MAX_UPLOAD_MB = int(os.environ.get("GEOAI_MAX_UPLOAD_MB", "1024"))
MAX_POINTS = 100_000  # ponytail: points sent to the browser are capped; tile server if datasets grow

app = FastAPI(title="GeoAI Tools")


# ================== IN-MEMORY STORES ==================
# ponytail: single-process in-memory store, cleared on restart; move to disk/object storage for multi-user use
class Stored:
    def __init__(self, name, data):
        self.id = uuid.uuid4().hex[:12]
        self.name = os.path.basename(name)
        self.ext = os.path.splitext(self.name)[1][1:].lower()
        self.kind = next((k for k, exts in KINDS.items() if self.ext in exts), None)
        self.data = data
        self.previews = {}  # cached display renders
        self.meta = {}

    def open(self):
        f = io.BytesIO(self.data)
        f.name = self.name
        return f

    def info(self):
        return {"id": self.id, "name": self.name, "ext": self.ext, "kind": self.kind,
                "size": len(self.data), "meta": self.meta}


FILES: "OrderedDict[str, Stored]" = OrderedDict()
RESULTS: "OrderedDict[str, tuple]" = OrderedDict()  # id -> (bytes, filename, media type)
RASTER_CACHE: "OrderedDict[str, tuple]" = OrderedDict()  # file id -> (wgs84 array, bounds)


def get_file(file_id, kinds=None):
    f = FILES.get(file_id)
    if f is None:
        raise HTTPException(404, "File not found. It may have been removed or the server restarted.")
    if kinds and f.kind not in kinds:
        raise HTTPException(400, f"{f.name} is not a {' or '.join(kinds)} file")
    return f


def add_result(data, filename, media):
    rid = uuid.uuid4().hex[:12]
    RESULTS[rid] = (data, filename, media)
    while len(RESULTS) > 40:
        RESULTS.popitem(last=False)
    return {"label": filename, "url": f"/api/results/{rid}"}


def stem(f):
    return os.path.splitext(f.name)[0]


@app.exception_handler(ValueError)
async def value_error(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ================== FILES ==================
def describe(f: Stored):
    if f.kind == "raster":
        with MemoryFile(f.data) as mem, mem.open() as src:
            # pixel-coordinate transforms (identity or y-flipped) carry no real location
            georef = src.crs is not None and src.transform not in (Affine.identity(), Affine(1, 0, 0, 0, -1, 0))
            f.meta = {"width": src.width, "height": src.height, "bands": src.count, "dtype": src.dtypes[0],
                      "crs": str(src.crs) if src.crs else None, "georeferenced": bool(georef),
                      "alpha": ColorInterp.alpha in src.colorinterp}
    elif f.kind == "image":
        with Image.open(f.open()) as im:
            f.meta = {"width": im.width, "height": im.height, "mode": im.mode}
    elif f.kind == "table":
        df = pd.read_csv(f.open())
        f.meta = {"rows": len(df), "columns": [str(c) for c in df.columns]}
    elif f.kind == "vector":
        f.meta = {"features": len(read_vector(f))}


def read_vector(f: Stored):
    import geopandas as gpd
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f.name)
        with open(path, "wb") as fh:
            fh.write(f.data)
        gdf = gpd.read_file(path)
    return gdf.to_crs(4326) if gdf.crs else gdf


@app.post("/api/files")
async def upload(files: list[UploadFile] = File(...)):
    out = []
    for up in files:
        chunks, size = [], 0
        while chunk := await up.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                raise HTTPException(413, f"{up.filename} is larger than {MAX_UPLOAD_MB} MB")
            chunks.append(chunk)
        f = Stored(up.filename or "upload", b"".join(chunks))
        if f.kind is None:
            out.append({"name": f.name, "error": f"Unsupported format .{f.ext}"})
            continue
        try:
            describe(f)
        except Exception as e:
            out.append({"name": f.name, "error": f"Could not read file: {e}"})
            continue
        FILES[f.id] = f
        out.append(f.info())
    return out


@app.get("/api/files")
def list_files():
    return [f.info() for f in FILES.values()]


@app.delete("/api/files/{file_id}")
def delete_file(file_id: str):
    FILES.pop(file_id, None)
    RASTER_CACHE.pop(file_id, None)
    return {"ok": True}


def preview_rgba(f: Stored, max_size):
    """Downsampled RGBA uint8 preview; masked/nodata pixels are transparent."""
    key = ("rgba", max_size)
    if key not in f.previews:
        if f.kind == "raster":
            with MemoryFile(f.data) as mem, mem.open() as src:
                bands = [i + 1 for i, c in enumerate(src.colorinterp) if c != ColorInterp.alpha][:3]
                scale = max(1, -(-max(src.width, src.height) // max_size))
                shape = (src.height // scale, src.width // scale)
                arr = np.moveaxis(src.read(bands, out_shape=(len(bands), *shape), resampling=Resampling.average), 0, -1)
                alpha = np.where(src.dataset_mask(out_shape=shape) > 0, 255, 0).astype(np.uint8)
            rgb = prepare_for_display(arr if arr.shape[-1] == 3 else np.repeat(arr[..., :1], 3, axis=-1))
            f.previews[key] = np.dstack([rgb, alpha])
        else:
            img = Image.open(f.open()).convert("RGBA")
            img.thumbnail((max_size, max_size))
            f.previews[key] = np.array(img)
    return f.previews[key]


@app.get("/api/files/{file_id}/thumb")
def thumbnail(file_id: str, size: int = 480):
    f = get_file(file_id, ["raster", "image"])
    size = max(64, min(2048, size))
    if ("png", size) not in f.previews:
        f.previews[("png", size)] = png_bytes(preview_rgba(f, size))
    return Response(f.previews[("png", size)], media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/files/{file_id}/geojson")
def geojson(file_id: str):
    return Response(read_vector(get_file(file_id, ["vector"])).to_json(), media_type="application/geo+json")


@app.get("/api/results/{rid}")
def result(rid: str):
    if rid not in RESULTS:
        raise HTTPException(404, "Result expired; run the tool again")
    data, filename, media = RESULTS[rid]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:150]  # names come from user uploads
    return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{safe}"'})


@app.get("/api/config")
def config():
    sats = {k: {"description": v["description"], "red": v["red"], "nir": v["nir"],
                "bands": {str(b): d for b, d in v["bands"].items()}} for k, v in SATELLITE_PROFILES.items()}
    return {"satellites": sats, "rgb_mode": RGB_MODE, "landuse_methods": available_methods()}


# ================== NDVI ==================
class NdviReq(BaseModel):
    file_id: str
    satellite: str = RGB_MODE
    red: Optional[int] = None
    nir: Optional[int] = None


def compute_ndvi(req: NdviReq):
    """Returns (ndvi array, mode label, file)."""
    from landuse_classifier.classifier import load_image_for_classification
    f = get_file(req.file_id, ["raster", "image"])
    if req.satellite == RGB_MODE:
        img = load_image_for_classification(f.open())[0].astype(float)
        return calculate_ndvi(img[..., 0], img[..., 1]), "Proxy NDVI", f
    if f.kind != "raster":
        raise ValueError(f"{req.satellite} bands need a multispectral GeoTIFF; use RGB mode for {f.ext.upper()} images")
    prof = get_satellite_bands(req.satellite)
    red, nir = req.red or prof["red"], req.nir or prof["nir"]
    if not red or not nir:
        raise ValueError("Choose red and NIR band numbers")
    if max(red, nir) > f.meta["bands"]:
        raise ValueError(f"{f.name} has {f.meta['bands']} bands, but {req.satellite} NDVI needs bands "
                         f"{red} (red) and {nir} (NIR). Try RGB mode or a Custom band selection.")
    return calculate_ndvi(*load_bands_with_mask(f.open(), red, nir)), "Real NDVI", f


NDVI_CLASSES = [("Water / bare", -1.0, 0.0, "#c8553d"), ("Sparse", 0.0, 0.2, "#e8c547"),
                ("Moderate", 0.2, 0.5, "#8cc265"), ("Dense", 0.5, 1.01, "#1e7b3c")]


@app.post("/api/ndvi")
def ndvi(req: NdviReq):
    nd, mode, f = compute_ndvi(req)
    valid = nd[~np.isnan(nd)]
    if valid.size == 0:
        raise ValueError("No valid pixels")
    rgba = colorize(nd, "RdYlGn", -1, 1)
    counts, edges = np.histogram(valid, bins=40, range=(-1, 1))
    downloads = [add_result(png_bytes(rgba), f"ndvi_{stem(f)}.png", "image/png")]
    if mode == "Real NDVI":
        downloads.append(add_result(to_geotiff_bytes(nd.astype("float32"), f.open(), nodata=np.nan),
                                    f"ndvi_{stem(f)}.tif", "image/tiff"))
    warnings = []
    if mode == "Proxy NDVI":
        warnings.append("No NIR band: green is used as a stand-in, so values are much smaller than real NDVI.")
    return {
        "mode": mode, "image": png_data_url(rgba), "width": nd.shape[1], "height": nd.shape[0],
        "stats": {"mean": float(valid.mean()), "min": float(valid.min()), "max": float(valid.max()),
                  "valid_pct": 100 * valid.size / nd.size},
        "classes": [{"label": n, "range": f"{lo:g} – {min(hi, 1):g}", "color": c,
                     "pct": float(100 * ((valid >= lo) & (valid < hi)).mean())} for n, lo, hi, c in NDVI_CLASSES],
        "histogram": {"edges": edges.round(3).tolist(), "counts": counts.tolist()},
        "warnings": warnings, "downloads": downloads,
    }


# ================== CROP MONITORING ==================
class CropCsvReq(BaseModel):
    file_id: str
    date_col: Optional[str] = None
    ndvi_col: Optional[str] = None


@app.post("/api/crop/csv")
def crop_csv(req: CropCsvReq):
    df = pd.read_csv(get_file(req.file_id, ["table"]).open())
    columns = [str(c) for c in df.columns]
    date_col = req.date_col or find_column(df, ["date", "time", "timestamp", "datetime"])
    ndvi_col = req.ndvi_col or find_column(df, ["ndvi", "vegetation", "index", "value"])
    if not date_col or not ndvi_col:
        return {"columns": columns, "date_col": date_col, "ndvi_col": ndvi_col, "points": []}
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"),
                        "ndvi": pd.to_numeric(df[ndvi_col], errors="coerce")}).dropna()
    if out.empty:
        raise ValueError(f"No rows with a valid date in '{date_col}' and a number in '{ndvi_col}'")
    return {"columns": columns, "date_col": date_col, "ndvi_col": ndvi_col,
            "points": [{"date": d.strftime("%Y-%m-%d"), "ndvi": float(v)} for d, v in zip(out["date"], out["ndvi"])]}


class CropImageReq(NdviReq):
    date: str


@app.post("/api/crop/image")
def crop_image(req: CropImageReq):
    nd, mode, _ = compute_ndvi(req)
    return {"date": req.date, "ndvi": float(np.nanmean(nd)), "mode": mode}


class Point(BaseModel):
    date: str
    ndvi: float


class AnalyzeReq(BaseModel):
    points: list[Point]
    threshold: float = 0.5
    smooth_days: int = 7


@app.post("/api/crop/analyze")
def crop_analyze(req: AnalyzeReq):
    df = pd.DataFrame([p.model_dump() for p in req.points])
    if df.empty:
        return {"raw": [], "smooth": [], "stressed": []}
    df["date"] = pd.to_datetime(df["date"])
    df = df.groupby("date", as_index=False)["ndvi"].mean().sort_values("date")
    raw = [{"date": d.strftime("%Y-%m-%d"), "ndvi": round(float(v), 4)} for d, v in zip(df["date"], df["ndvi"])]
    smooth = []
    if req.smooth_days > 0 and len(df) > 1:
        daily = df.set_index("date")["ndvi"].asfreq("D").interpolate(method="time")
        rolled = daily.rolling(f"{req.smooth_days}D", min_periods=1).mean()
        smooth = [{"date": d.strftime("%Y-%m-%d"), "ndvi": round(float(v), 4)} for d, v in rolled.items()]
    # Stress is judged on the observed dates (smoothed value where available)
    lookup = {p["date"]: p["ndvi"] for p in smooth}
    stressed = [p | {"value": lookup.get(p["date"], p["ndvi"])} for p in raw
                if lookup.get(p["date"], p["ndvi"]) < req.threshold]
    return {"raw": raw, "smooth": smooth, "stressed": stressed}


# ================== LAND USE ==================
class LanduseReq(BaseModel):
    file_id: str
    method: str = "Spectral Clustering"
    water_threshold: float = -0.3
    veg_threshold: float = 0.3
    n_clusters: int = 6


@app.post("/api/landuse")
def landuse(req: LanduseReq):
    f = get_file(req.file_id, ["raster", "image"])
    if req.method not in available_methods():
        raise ValueError(f"Method not available: {req.method}")
    img, classified, names, colors, has_nir = classify(f.open(), req.method, req.water_threshold,
                                                       req.veg_threshold, max(2, min(10, req.n_clusters)))
    original = prepare_for_display(img)
    colored = create_colored_classification_map(classified, colors)
    overlay = (0.6 * original + 0.4 * colored).astype(np.uint8)

    ids, counts = np.unique(classified, return_counts=True)
    classes = [{"id": int(i), "name": names.get(int(i), f"Class {i}"), "color": "#%02x%02x%02x" % tuple(colors.get(int(i), [128] * 3)),
                "pixels": int(c), "pct": float(100 * c / classified.size)} for i, c in zip(ids, counts)]
    stats_csv = pd.DataFrame(classes)[["name", "pixels", "pct"]].rename(
        columns={"name": "class", "pct": "percentage"}).to_csv(index=False).encode()
    downloads = [add_result(png_bytes(colored), f"landuse_{stem(f)}.png", "image/png"),
                 add_result(stats_csv, f"landuse_{stem(f)}.csv", "text/csv")]
    if f.kind == "raster":
        downloads.insert(1, add_result(to_geotiff_bytes(classified.astype(np.uint8), f.open()),
                                       f"landuse_{stem(f)}.tif", "image/tiff"))
    warnings = []
    if req.method == "Simple NDVI-based" and not has_nir:
        warnings.append("No NIR band found: NDVI is approximated from red and green, so classes are rough. "
                        "Spectral Clustering suits RGB/grayscale imagery better.")
    return {"original": png_data_url(original), "classified": png_data_url(colored),
            "overlay": png_data_url(overlay), "classes": classes, "warnings": warnings, "downloads": downloads}


# ================== OBJECT DETECTION ==================
class DetectReq(BaseModel):
    file_id: str
    conf: float = 0.25
    iou: float = 0.45


@app.post("/api/detect")
def detect(req: DetectReq):
    from satellite_detection.detector import detect as run_detection
    f = get_file(req.file_id, ["raster", "image"])
    img, detections = run_detection(f.open(), req.conf, req.iou)
    counts = pd.Series([d["class_name"] for d in detections], dtype=object).value_counts()
    downloads = [add_result(png_bytes(img), f"detected_{stem(f)}.png", "image/png")]
    if detections:
        downloads.append(add_result(pd.DataFrame(detections).to_csv(index=False).encode(),
                                    f"detections_{stem(f)}.csv", "text/csv"))
    return {"image": png_data_url(img), "detections": detections,
            "counts": [{"name": n, "count": int(c)} for n, c in counts.items()], "downloads": downloads}


# ================== GPS + POLLUTION POINTS ==================
class PointsReq(BaseModel):
    file_id: str
    lat_col: Optional[str] = None
    lon_col: Optional[str] = None
    value_col: Optional[str] = None


POLLUTANT_NAMES = ["no2", "pm25", "pm2_5", "pm2.5", "pm10", "co", "so2", "o3", "aqi", "value", "concentration"]


def load_points(req: PointsReq, value_candidates=None):
    df = pd.read_csv(get_file(req.file_id, ["table"]).open())
    columns = [str(c) for c in df.columns]
    numeric = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    lat = req.lat_col or find_column(df, LAT_NAMES)
    lon = req.lon_col or find_column(df, LON_NAMES)
    value = req.value_col or (find_column(df, value_candidates) if value_candidates else None)
    base = {"columns": columns, "numeric_columns": numeric, "lat_col": lat, "lon_col": lon, "value_col": value}
    if not lat or not lon:
        return base, None
    return base, clean_points(df, lat, lon, value)


def stats_of(values):
    counts, edges = np.histogram(values, bins=24)
    return ({"count": int(values.size), "min": float(values.min()), "max": float(values.max()),
             "mean": float(values.mean()), "std": float(values.std())},
            {"edges": edges.round(4).tolist(), "counts": counts.tolist()})


@app.post("/api/gps")
def gps(req: PointsReq):
    base, pts = load_points(req)
    if pts is None:
        return base | {"points": []}
    cols = [base["lat_col"], base["lon_col"]] + ([base["value_col"]] if base["value_col"] else [])
    return base | {"points": pts[cols].head(MAX_POINTS).round(6).values.tolist(), "total": len(pts)}


@app.post("/api/pollution/csv")
def pollution_csv(req: PointsReq):
    base, pts = load_points(req, POLLUTANT_NAMES)
    base["pollutants"] = [c for c in base["numeric_columns"] if c not in (base["lat_col"], base["lon_col"])]
    if pts is None or not base["value_col"]:
        return base | {"points": []}
    stats, hist = stats_of(pts[base["value_col"]].to_numpy(float))
    return base | {"points": pts[[base["lat_col"], base["lon_col"], base["value_col"]]].head(MAX_POINTS).values.tolist(),
                   "total": len(pts), "stats": stats, "histogram": hist}


class RasterReq(BaseModel):
    file_id: str
    vmin: Optional[float] = None
    vmax: Optional[float] = None


@app.post("/api/pollution/raster")
def pollution_raster(req: RasterReq):
    f = get_file(req.file_id, ["raster"])
    if f.id not in RASTER_CACHE:
        with MemoryFile(f.data) as mem, mem.open() as src:
            data, b = read_band_wgs84(src)
            name = (src.descriptions[0] if src.descriptions and src.descriptions[0] else None) or src.tags().get("pollutant")
        RASTER_CACHE[f.id] = (data, b, name)
        while len(RASTER_CACHE) > 8:
            RASTER_CACHE.popitem(last=False)
    data, b, name = RASTER_CACHE[f.id]
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        raise ValueError("Raster has no valid pixels")
    lo = valid.min() if req.vmin is None else req.vmin
    hi = valid.max() if req.vmax is None else req.vmax
    shown = np.where((data >= lo) & (data <= hi), data, np.nan)
    rgba = colorize(shown, "inferno", lo, hi)
    stats, hist = stats_of(valid)
    warnings = []
    if not f.meta.get("georeferenced"):
        warnings.append("This raster has no real-world georeferencing, so its map position is not meaningful.")
    return {"image": png_data_url(rgba, max_size=2048), "bounds": [[b.bottom, b.left], [b.top, b.right]],
            "pollutant": name, "stats": stats, "histogram": hist, "range": [float(lo), float(hi)],
            "warnings": warnings,
            "downloads": [add_result(png_bytes(rgba), f"{name or 'pollution'}_{stem(f)}.png", "image/png")]}


# ================== GEOREFERENCING ==================
class Gcp(BaseModel):
    col: float
    row: float
    lon: float
    lat: float


class GeorefReq(BaseModel):
    file_id: str
    gcps: list[Gcp]
    save: bool = False


@app.post("/api/georef")
def georef(req: GeorefReq):
    f = get_file(req.file_id, ["raster", "image"])
    w, h = f.meta["width"], f.meta["height"]
    crs, transform, residuals = fit_gcps([(g.col, g.row, g.lon, g.lat) for g in req.gcps])
    preview, b = warp_preview(preview_rgba(f, 1024), crs, transform, w, h)
    out = {"crs": crs, "rmse_m": float(np.sqrt(np.mean(residuals ** 2))), "residuals_m": residuals.round(2).tolist(),
           "pixel_size_m": float(np.sqrt(abs(transform.determinant))),
           "preview": {"image": png_data_url(preview, max_size=2048), "bounds": [[b.bottom, b.left], [b.top, b.right]]}}
    if req.save:
        data = f.data if f.kind == "raster" else np.array(Image.open(f.open()).convert("RGB"))
        tif = georeferenced_geotiff(data, crs, transform)
        new = Stored(f"{stem(f).removesuffix('_georef')}_georef.tif", tif)
        describe(new)
        FILES[new.id] = new
        out["file"] = new.info()
        out["downloads"] = [add_result(tif, new.name, "image/tiff")]
    return out


# Static frontend (registered last so /api routes win)
app.mount("/", StaticFiles(directory=os.path.join(HERE, "frontend"), html=True), name="frontend")
