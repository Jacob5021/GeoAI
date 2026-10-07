"""GeoAI Tools: FastAPI backend + static web frontend.

Run:  uvicorn server:app --port 8599
"""
import io
import os
import tempfile
import time
from collections import OrderedDict, deque
from typing import Annotated, Optional

import numpy as np
import pandas as pd
from affine import Affine
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from rasterio.enums import ColorInterp, Resampling
from rasterio.io import MemoryFile

import storage
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
storage.init()
COOKIE = "geoai_session"
SECURE_COOKIES = os.environ.get("GEOAI_SECURE_COOKIES", "false").lower() == "true"  # set true behind HTTPS
ALLOW_SIGNUP = os.environ.get("GEOAI_ALLOW_SIGNUP", "true").lower() == "true"


@app.exception_handler(ValueError)
async def value_error(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ================== AUTH ==================
def current_user(request: Request):
    user = storage.user_for_session(request.cookies.get(COOKIE))
    if not user:
        raise HTTPException(401, "Please sign in")
    return user


User = Annotated[dict, Depends(current_user)]

# ponytail: in-memory login throttle per process; use a shared store if running several workers
FAILED_LOGINS: "dict[str, deque]" = {}


def throttle(key):
    attempts = FAILED_LOGINS.setdefault(key, deque())
    while attempts and attempts[0] < time.time() - 900:
        attempts.popleft()
    if len(attempts) >= 10:
        raise HTTPException(429, "Too many failed sign-in attempts. Try again in 15 minutes.")
    return attempts


class Credentials(BaseModel):
    username: str
    password: str


def signed_in(user):
    resp = JSONResponse(user)
    resp.set_cookie(COOKIE, storage.create_session(user["id"]), max_age=storage.SESSION_SECONDS,
                    httponly=True, samesite="lax", secure=SECURE_COOKIES, path="/")
    return resp


@app.post("/api/auth/register")
def register(body: Credentials):
    if not ALLOW_SIGNUP:
        raise HTTPException(403, "Sign-up is disabled on this server")
    return signed_in(storage.create_user(body.username.strip(), body.password))


@app.post("/api/auth/login")
def login(body: Credentials, request: Request):
    attempts = throttle(f"{request.client.host if request.client else ''}:{body.username.lower()}")
    user = storage.authenticate(body.username.strip(), body.password)
    if not user:
        attempts.append(time.time())
        raise HTTPException(401, "Wrong username or password")
    return signed_in(user)


@app.post("/api/auth/logout")
def logout(request: Request):
    storage.delete_session(request.cookies.get(COOKIE))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, path="/")
    return resp


@app.get("/api/auth/me")
def me(user: User):
    return user


class PasswordChange(BaseModel):
    current: str
    new: str


@app.post("/api/auth/password")
def change_password(body: PasswordChange, user: User):
    if not storage.authenticate(user["username"], body.current):
        raise HTTPException(400, "Current password is wrong")
    storage.set_password(user["id"], body.new)
    return signed_in(user)


# ================== DATASETS ==================
# Decoded bytes/renders are cached in memory; the source of truth is storage (SQLite + data/ folders)
BYTES_CACHE: "OrderedDict[str, bytes]" = OrderedDict()
PREVIEW_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
RASTER_CACHE: "OrderedDict[str, tuple]" = OrderedDict()  # file id -> (wgs84 array, bounds, name)


def cache_put(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


class Dataset:
    """A stored dataset as the tools see it."""

    def __init__(self, row):
        self.row = row
        self.id, self.name, self.ext, self.kind, self.meta = row["id"], row["name"], row["ext"], row["kind"], row["meta"]

    @property
    def data(self):
        if self.id in BYTES_CACHE:
            BYTES_CACHE.move_to_end(self.id)
            return BYTES_CACHE[self.id]
        with open(storage.file_path(self.row), "rb") as fh:
            return cache_put(BYTES_CACHE, self.id, fh.read(), 3)

    def open(self):
        f = io.BytesIO(self.data)
        f.name = self.name
        return f


def kind_of(ext):
    return next((k for k, exts in KINDS.items() if ext in exts), None)


def get_file(user, file_id, kinds=None):
    row = storage.get_file(user, file_id)
    if row is None:
        raise HTTPException(404, "Dataset not found")
    f = Dataset(row)
    if kinds and f.kind not in kinds:
        raise HTTPException(400, f"{f.name} is not a {' or '.join(kinds)} file")
    return f


def stem(f):
    return os.path.splitext(f.name)[0]


def describe(name, kind, data):
    f = io.BytesIO(data)
    f.name = name
    if kind == "raster":
        with MemoryFile(data) as mem, mem.open() as src:
            # pixel-coordinate transforms (identity or y-flipped) carry no real location
            georef = src.crs is not None and src.transform not in (Affine.identity(), Affine(1, 0, 0, 0, -1, 0))
            return {"width": src.width, "height": src.height, "bands": src.count, "dtype": src.dtypes[0],
                    "crs": str(src.crs) if src.crs else None, "georeferenced": bool(georef),
                    "alpha": ColorInterp.alpha in src.colorinterp}
    if kind == "image":
        with Image.open(f) as im:
            return {"width": im.width, "height": im.height, "mode": im.mode}
    if kind == "table":
        df = pd.read_csv(f)
        return {"rows": len(df), "columns": [str(c) for c in df.columns]}
    return {"features": len(read_vector_bytes(name, data))}


def read_vector_bytes(name, data):
    import geopandas as gpd
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, storage.safe_name(name))
        with open(path, "wb") as fh:
            fh.write(data)
        gdf = gpd.read_file(path)
    return gdf.to_crs(4326) if gdf.crs else gdf


def add_dataset(user, name, data):
    """Validate, describe and store a new dataset. Returns its info or raises ValueError."""
    ext = os.path.splitext(name)[1][1:].lower()
    kind = kind_of(ext)
    if kind is None:
        raise ValueError(f"Unsupported format .{ext}")
    try:
        meta = describe(name, kind, data)
    except Exception as e:
        raise ValueError(f"Could not read file: {e}") from e
    return storage.save_file(user, name, ext, kind, data, meta)


def file_info(row, results=()):
    return {k: row[k] for k in ("id", "name", "ext", "kind", "size", "meta", "created_at")} | {"results": list(results)}


@app.post("/api/files")
async def upload(user: User, files: list[UploadFile] = File(...)):
    out = []
    for up in files:
        chunks, size = [], 0
        while chunk := await up.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                raise HTTPException(413, f"{up.filename} is larger than {MAX_UPLOAD_MB} MB")
            chunks.append(chunk)
        name = os.path.basename(up.filename or "upload")
        try:
            out.append(file_info(add_dataset(user, name, b"".join(chunks))))
        except ValueError as e:
            out.append({"name": name, "error": str(e)})
    return out


@app.get("/api/files")
def list_files(user: User):
    by_file = {}
    for r in storage.list_results(user):
        by_file.setdefault(r["file_id"], []).append(r)
    return [file_info(f, by_file.get(f["id"], [])) for f in storage.list_files(user)]


@app.delete("/api/files/{file_id}")
def delete_file(file_id: str, user: User):
    if not storage.delete_file(user, file_id):
        raise HTTPException(404, "Dataset not found")
    for cache in (BYTES_CACHE, RASTER_CACHE):
        cache.pop(file_id, None)
    return {"ok": True}


@app.get("/api/files/{file_id}/download")
def download_file(file_id: str, user: User):
    f = get_file(user, file_id)
    return FileResponse(storage.file_path(f.row), filename=f.row["stored_name"])


def preview_rgba(f: Dataset, max_size):
    """Downsampled RGBA uint8 preview; masked/nodata pixels are transparent."""
    key = (f.id, "rgba", max_size)
    if key in PREVIEW_CACHE:
        return PREVIEW_CACHE[key]
    if f.kind == "raster":
        with MemoryFile(f.data) as mem, mem.open() as src:
            bands = [i + 1 for i, c in enumerate(src.colorinterp) if c != ColorInterp.alpha][:3]
            scale = max(1, -(-max(src.width, src.height) // max_size))
            shape = (src.height // scale, src.width // scale)
            arr = np.moveaxis(src.read(bands, out_shape=(len(bands), *shape), resampling=Resampling.average), 0, -1)
            alpha = np.where(src.dataset_mask(out_shape=shape) > 0, 255, 0).astype(np.uint8)
        rgb = prepare_for_display(arr if arr.shape[-1] == 3 else np.repeat(arr[..., :1], 3, axis=-1))
        out = np.dstack([rgb, alpha])
    else:
        img = Image.open(f.open()).convert("RGBA")
        img.thumbnail((max_size, max_size))
        out = np.array(img)
    return cache_put(PREVIEW_CACHE, key, out, 32)


@app.get("/api/files/{file_id}/thumb")
def thumbnail(file_id: str, user: User, size: int = 480):
    f = get_file(user, file_id, ["raster", "image"])
    size = max(64, min(2048, size))
    key = (f.id, "png", size)
    png = PREVIEW_CACHE[key] if key in PREVIEW_CACHE else cache_put(PREVIEW_CACHE, key, png_bytes(preview_rgba(f, size)), 32)
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/files/{file_id}/geojson")
def geojson(file_id: str, user: User):
    f = get_file(user, file_id, ["vector"])
    return Response(read_vector_bytes(f.name, f.data).to_json(), media_type="application/geo+json")


# ================== SAVED RESULTS ==================
def run_saved(user, f, tool, params, force, compute):
    """Return the stored result for identical settings, or compute, store next to the dataset and return.

    compute() -> (payload, outputs [(bytes, filename, label, media)], summary)."""
    if not force:
        rid = storage.find_result(user, f.id, tool, storage.params_key(params))
        payload = rid and storage.load_payload(user, rid)
        if payload:
            payload["result"]["cached"] = True
            return payload
    payload, outputs, summary = compute()
    return storage.save_result(user, f.row, tool, params, summary, payload, outputs)


def settings(req, *drop):
    return req.model_dump(exclude={"file_id", "force", *drop})


@app.get("/api/results")
def list_results(user: User, file_id: Optional[str] = None):
    return storage.list_results(user, file_id)


@app.get("/api/results/{result_id}")
def get_result(result_id: str, user: User):
    payload = storage.load_payload(user, result_id)
    if payload is None:
        raise HTTPException(404, "Result not found")
    payload["result"]["cached"] = True
    return payload


@app.delete("/api/results/{result_id}")
def delete_result(result_id: str, user: User):
    if not storage.delete_result(user, result_id):
        raise HTTPException(404, "Result not found")
    return {"ok": True}


@app.get("/api/outputs/{output_id}")
def output(output_id: str, user: User):
    found = storage.get_output(user, output_id)
    if not found:
        raise HTTPException(404, "File not found")
    path, filename, media = found
    return FileResponse(path, media_type=media, filename=filename)


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
    force: bool = False


def compute_ndvi(f: Dataset, req: NdviReq):
    """Returns (ndvi array, mode label)."""
    from landuse_classifier.classifier import load_image_for_classification
    if req.satellite == RGB_MODE:
        img = load_image_for_classification(f.open())[0].astype(float)
        return calculate_ndvi(img[..., 0], img[..., 1]), "Proxy NDVI"
    if f.kind != "raster":
        raise ValueError(f"{req.satellite} bands need a multispectral GeoTIFF; use RGB mode for {f.ext.upper()} images")
    prof = get_satellite_bands(req.satellite)
    red, nir = req.red or prof["red"], req.nir or prof["nir"]
    if not red or not nir:
        raise ValueError("Choose red and NIR band numbers")
    if max(red, nir) > f.meta["bands"]:
        raise ValueError(f"{f.name} has {f.meta['bands']} bands, but {req.satellite} NDVI needs bands "
                         f"{red} (red) and {nir} (NIR). Try RGB mode or a Custom band selection.")
    return calculate_ndvi(*load_bands_with_mask(f.open(), red, nir)), "Real NDVI"


NDVI_CLASSES = [("Water / bare", -1.0, 0.0, "#c8553d"), ("Sparse", 0.0, 0.2, "#e8c547"),
                ("Moderate", 0.2, 0.5, "#8cc265"), ("Dense", 0.5, 1.01, "#1e7b3c")]


@app.post("/api/ndvi")
def ndvi(req: NdviReq, user: User):
    f = get_file(user, req.file_id, ["raster", "image"])

    def compute():
        nd, mode = compute_ndvi(f, req)
        valid = nd[~np.isnan(nd)]
        if valid.size == 0:
            raise ValueError("No valid pixels")
        rgba = colorize(nd, "RdYlGn", -1, 1)
        counts, edges = np.histogram(valid, bins=40, range=(-1, 1))
        outputs = [(png_bytes(rgba), f"ndvi_{stem(f)}.png", "NDVI map (PNG)", "image/png")]
        if mode == "Real NDVI":
            outputs.append((to_geotiff_bytes(nd.astype("float32"), f.open(), nodata=np.nan),
                            f"ndvi_{stem(f)}.tif", "NDVI GeoTIFF", "image/tiff"))
        stats = {"mean": float(valid.mean()), "min": float(valid.min()), "max": float(valid.max()),
                 "valid_pct": 100 * valid.size / nd.size}
        payload = {
            "mode": mode, "image": png_data_url(rgba), "width": nd.shape[1], "height": nd.shape[0], "stats": stats,
            "classes": [{"label": n, "range": f"{lo:g} – {min(hi, 1):g}", "color": c,
                         "pct": float(100 * ((valid >= lo) & (valid < hi)).mean())} for n, lo, hi, c in NDVI_CLASSES],
            "histogram": {"edges": edges.round(3).tolist(), "counts": counts.tolist()},
            "warnings": ["No NIR band: green is used as a stand-in, so values are much smaller than real NDVI."]
            if mode == "Proxy NDVI" else [],
        }
        return payload, outputs, {"mode": mode, "mean": stats["mean"]}

    return run_saved(user, f, "ndvi", settings(req), req.force, compute)


# ================== CROP MONITORING ==================
class CropCsvReq(BaseModel):
    file_id: str
    date_col: Optional[str] = None
    ndvi_col: Optional[str] = None


@app.post("/api/crop/csv")
def crop_csv(req: CropCsvReq, user: User):
    df = pd.read_csv(get_file(user, req.file_id, ["table"]).open())
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
def crop_image(req: CropImageReq, user: User):
    f = get_file(user, req.file_id, ["raster", "image"])

    def compute():
        nd, mode = compute_ndvi(f, req)
        point = {"date": req.date, "ndvi": float(np.nanmean(nd)), "mode": mode}
        return point, [], point

    return run_saved(user, f, "crop", settings(req), req.force, compute)


class Point(BaseModel):
    date: str
    ndvi: float


class AnalyzeReq(BaseModel):
    points: list[Point]
    threshold: float = 0.5
    smooth_days: int = 7


@app.post("/api/crop/analyze")
def crop_analyze(req: AnalyzeReq, user: User):
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
    force: bool = False


@app.post("/api/landuse")
def landuse(req: LanduseReq, user: User):
    f = get_file(user, req.file_id, ["raster", "image"])
    if req.method not in available_methods():
        raise ValueError(f"Method not available: {req.method}")
    # Only the settings that affect this method identify the result
    params = {"method": req.method} | ({"water_threshold": req.water_threshold, "veg_threshold": req.veg_threshold}
                                       if req.method == "Simple NDVI-based" else
                                       {"n_clusters": max(2, min(10, req.n_clusters))} if req.method == "Spectral Clustering" else {})

    def compute():
        img, classified, names, colors, has_nir = classify(f.open(), req.method, req.water_threshold,
                                                           req.veg_threshold, params.get("n_clusters", 6))
        original = prepare_for_display(img)
        colored = create_colored_classification_map(classified, colors)
        overlay = (0.6 * original + 0.4 * colored).astype(np.uint8)
        ids, counts = np.unique(classified, return_counts=True)
        classes = [{"id": int(i), "name": names.get(int(i), f"Class {i}"),
                    "color": "#%02x%02x%02x" % tuple(colors.get(int(i), [128] * 3)),
                    "pixels": int(c), "pct": float(100 * c / classified.size)} for i, c in zip(ids, counts)]
        stats_csv = pd.DataFrame(classes)[["name", "pixels", "pct"]].rename(
            columns={"name": "class", "pct": "percentage"}).to_csv(index=False).encode()
        outputs = [(png_bytes(colored), f"landuse_{stem(f)}.png", "Class map (PNG)", "image/png")]
        if f.kind == "raster":
            outputs.append((to_geotiff_bytes(classified.astype(np.uint8), f.open()),
                            f"landuse_{stem(f)}.tif", "Class GeoTIFF", "image/tiff"))
        outputs.append((stats_csv, f"landuse_{stem(f)}.csv", "Statistics (CSV)", "text/csv"))
        payload = {"method": req.method, "original": png_data_url(original), "classified": png_data_url(colored),
                   "overlay": png_data_url(overlay), "classes": classes,
                   "warnings": ["No NIR band found: NDVI is approximated from red and green, so classes are rough. "
                                "Spectral Clustering suits RGB/grayscale imagery better."]
                   if req.method == "Simple NDVI-based" and not has_nir else []}
        top = max(classes, key=lambda c: c["pct"])
        return payload, outputs, {"method": req.method, "classes": len(classes), "largest": f"{top['name']} {top['pct']:.0f}%"}

    return run_saved(user, f, "landuse", params, req.force, compute)


# ================== OBJECT DETECTION ==================
class DetectReq(BaseModel):
    file_id: str
    conf: float = 0.25
    iou: float = 0.45
    force: bool = False


@app.post("/api/detect")
def detect(req: DetectReq, user: User):
    from satellite_detection.detector import detect as run_detection
    f = get_file(user, req.file_id, ["raster", "image"])

    def compute():
        img, detections = run_detection(f.open(), req.conf, req.iou)
        counts = pd.Series([d["class_name"] for d in detections], dtype=object).value_counts()
        outputs = [(png_bytes(img), f"detected_{stem(f)}.png", "Annotated image (PNG)", "image/png")]
        if detections:
            outputs.append((pd.DataFrame(detections).to_csv(index=False).encode(),
                            f"detections_{stem(f)}.csv", "Detections (CSV)", "text/csv"))
        payload = {"image": png_data_url(img), "detections": detections,
                   "counts": [{"name": n, "count": int(c)} for n, c in counts.items()]}
        return payload, outputs, {"objects": len(detections)}

    return run_saved(user, f, "detect", settings(req), req.force, compute)


# ================== GPS + POLLUTION POINTS ==================
class PointsReq(BaseModel):
    file_id: str
    lat_col: Optional[str] = None
    lon_col: Optional[str] = None
    value_col: Optional[str] = None
    force: bool = False


POLLUTANT_NAMES = ["no2", "pm25", "pm2_5", "pm2.5", "pm10", "co", "so2", "o3", "aqi", "value", "concentration"]


def load_points(f, req: PointsReq, value_candidates=None):
    df = pd.read_csv(f.open())
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
def gps(req: PointsReq, user: User):
    f = get_file(user, req.file_id, ["table"])
    base, pts = load_points(f, req)
    if pts is None:  # columns not detected: ask, don't store
        return base | {"points": []}
    params = {"lat_col": base["lat_col"], "lon_col": base["lon_col"], "value_col": base["value_col"]}

    def compute():
        cols = [base["lat_col"], base["lon_col"]] + ([base["value_col"]] if base["value_col"] else [])
        payload = base | {"points": pts[cols].head(MAX_POINTS).round(6).values.tolist(), "total": len(pts)}
        return payload, [], {"points": len(pts)}

    return run_saved(user, f, "gps", params, req.force, compute)


@app.post("/api/pollution/csv")
def pollution_csv(req: PointsReq, user: User):
    f = get_file(user, req.file_id, ["table"])
    base, pts = load_points(f, req, POLLUTANT_NAMES)
    base["pollutants"] = [c for c in base["numeric_columns"] if c not in (base["lat_col"], base["lon_col"])]
    if pts is None or not base["value_col"]:
        return base | {"type": "csv", "points": []}
    params = {"lat_col": base["lat_col"], "lon_col": base["lon_col"], "value_col": base["value_col"]}

    def compute():
        stats, hist = stats_of(pts[base["value_col"]].to_numpy(float))
        payload = base | {"type": "csv", "total": len(pts), "stats": stats, "histogram": hist,
                          "points": pts[[base["lat_col"], base["lon_col"], base["value_col"]]].head(MAX_POINTS).values.tolist()}
        return payload, [], {"pollutant": base["value_col"], "mean": stats["mean"], "max": stats["max"]}

    return run_saved(user, f, "pollution", params, req.force, compute)


class RasterReq(BaseModel):
    file_id: str
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    force: bool = False


def raster_wgs84(f):
    if f.id not in RASTER_CACHE:
        with MemoryFile(f.data) as mem, mem.open() as src:
            data, b = read_band_wgs84(src)
            name = (src.descriptions[0] if src.descriptions and src.descriptions[0] else None) or src.tags().get("pollutant")
        cache_put(RASTER_CACHE, f.id, (data, b, name), 8)
    return RASTER_CACHE[f.id]


@app.post("/api/pollution/raster")
def pollution_raster(req: RasterReq, user: User):
    f = get_file(user, req.file_id, ["raster"])

    def compute():
        data, b, name = raster_wgs84(f)
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            raise ValueError("Raster has no valid pixels")
        lo = valid.min() if req.vmin is None else req.vmin
        hi = valid.max() if req.vmax is None else req.vmax
        rgba = colorize(np.where((data >= lo) & (data <= hi), data, np.nan), "inferno", lo, hi)
        stats, hist = stats_of(valid)
        payload = {"type": "raster", "image": png_data_url(rgba, max_size=2048),
                   "bounds": [[b.bottom, b.left], [b.top, b.right]], "pollutant": name, "stats": stats,
                   "histogram": hist, "range": [float(lo), float(hi)],
                   "warnings": [] if f.meta.get("georeferenced") else
                   ["This raster has no real-world georeferencing, so its map position is not meaningful."]}
        outputs = [(png_bytes(rgba), f"{name or 'pollution'}_{stem(f)}.png", "Raster map (PNG)", "image/png")]
        return payload, outputs, {"mean": stats["mean"], "max": stats["max"]}

    if req.vmin is not None or req.vmax is not None:  # live display-range tweaks aren't stored
        payload, _, _ = compute()
        return payload
    return run_saved(user, f, "pollution", {"type": "raster"}, req.force, compute)


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
def georef(req: GeorefReq, user: User):
    f = get_file(user, req.file_id, ["raster", "image"])
    w, h = f.meta["width"], f.meta["height"]
    crs, transform, residuals = fit_gcps([(g.col, g.row, g.lon, g.lat) for g in req.gcps])
    preview, b = warp_preview(preview_rgba(f, 1024), crs, transform, w, h)
    out = {"crs": crs, "rmse_m": float(np.sqrt(np.mean(residuals ** 2))), "residuals_m": residuals.round(2).tolist(),
           "pixel_size_m": float(np.sqrt(abs(transform.determinant))), "gcps": [g.model_dump() for g in req.gcps],
           "preview": {"image": png_data_url(preview, max_size=2048), "bounds": [[b.bottom, b.left], [b.top, b.right]]}}
    if not req.save:  # live fits while placing points aren't stored
        return out
    data = f.data if f.kind == "raster" else np.array(Image.open(f.open()).convert("RGB"))
    tif = georeferenced_geotiff(data, crs, transform)
    new = add_dataset(user, f"{stem(f).removesuffix('_georef')}_georef.tif", tif)
    out["file"] = file_info(new)
    summary = {"rmse_m": out["rmse_m"], "crs": crs, "points": len(req.gcps), "saved_as": new["name"]}
    # The GeoTIFF itself becomes a new dataset in the library; the result keeps a copy for download
    return storage.save_result(user, f.row, "georef", {"gcps": out["gcps"]}, summary, out,
                               [(tif, new["name"], "Georeferenced GeoTIFF", "image/tiff")])


# Static frontend (registered last so /api routes win)
app.mount("/", StaticFiles(directory=os.path.join(HERE, "frontend"), html=True), name="frontend")
