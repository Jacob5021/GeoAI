"""End-to-end API checks: upload synthetic files and call every tool endpoint."""
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from server import app

client = TestClient(app)


def s2_tif():
    """8-band uint16 UTM raster, vegetated top half."""
    bands = (np.random.default_rng(0).random((8, 40, 40)) * 2000 + 500).astype("uint16")
    bands[7, :20] += 4000
    mem = MemoryFile()
    with mem.open(driver="GTiff", height=40, width=40, count=8, dtype="uint16", nodata=0,
                  crs="EPSG:32643", transform=from_origin(780000, 1440000, 10, 10)) as dst:
        dst.write(bands)
    return mem.read()


def png():
    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(1).random((32, 32, 3)) * 255).astype("uint8")).save(buf, "PNG")
    return buf.getvalue()


def upload(name, data):
    r = client.post("/api/files", files=[("files", (name, data))])
    assert r.status_code == 200 and "id" in r.json()[0], r.text
    return r.json()[0]


def test_tools_end_to_end():
    tif, img = upload("s2.tif", s2_tif()), upload("rgb.png", png())
    gps = upload("gps.csv", ("Latitude,Longitude,w\n" + "\n".join(f"{12.9 + i / 1e3},{77.6 + i / 1e3},{i}" for i in range(30))).encode())
    poll = upload("air.csv", ("lat,lon,no2\n" + "\n".join(f"{12 + i / 100},{77 + i / 100},{20 + i}" for i in range(20))).encode())
    series = upload("ndvi.csv", b"day,greenness\n2026-01-01,0.6\n2026-01-01,0.4\n2026-01-09,0.3\n2026-01-20,0.7\n")
    assert tif["meta"]["georeferenced"] and tif["meta"]["bands"] == 8

    assert client.get(f"/api/files/{tif['id']}/thumb").headers["content-type"] == "image/png"

    r = client.post("/api/ndvi", json={"file_id": tif["id"], "satellite": "Sentinel-2"}).json()
    assert r["mode"] == "Real NDVI" and len(r["downloads"]) == 2 and abs(sum(c["pct"] for c in r["classes"]) - 100) < 0.01
    assert client.get(r["downloads"][1]["url"]).content[:2] in (b"II", b"MM")  # GeoTIFF magic
    r = client.post("/api/ndvi", json={"file_id": img["id"]}).json()
    assert r["mode"] == "Proxy NDVI" and r["warnings"]
    bad = client.post("/api/ndvi", json={"file_id": img["id"], "satellite": "Sentinel-2"})
    assert bad.status_code == 400 and "GeoTIFF" in bad.json()["detail"]

    for method in ["Simple NDVI-based", "Spectral Clustering"]:
        r = client.post("/api/landuse", json={"file_id": tif["id"], "method": method, "n_clusters": 4}).json()
        assert abs(sum(c["pct"] for c in r["classes"]) - 100) < 0.01 and len(r["downloads"]) == 3, r

    r = client.post("/api/detect", json={"file_id": img["id"]}).json()
    assert r["image"].startswith("data:image/png") and isinstance(r["detections"], list)

    r = client.post("/api/gps", json={"file_id": gps["id"]}).json()
    assert (r["lat_col"], r["lon_col"]) == ("Latitude", "Longitude") and len(r["points"]) == 30

    r = client.post("/api/pollution/csv", json={"file_id": poll["id"]}).json()
    assert r["value_col"] == "no2" and r["stats"]["max"] == 39
    r = client.post("/api/pollution/raster", json={"file_id": tif["id"]}).json()
    (s, w), (n, e) = r["bounds"]
    assert 77 < w < e < 78 and 12 < s < n < 14  # reprojected to degrees

    r = client.post("/api/crop/csv", json={"file_id": series["id"]}).json()
    assert r["points"] == [] and r["columns"] == ["day", "greenness"]  # undetected -> ask for columns
    r = client.post("/api/crop/csv", json={"file_id": series["id"], "date_col": "day", "ndvi_col": "greenness"}).json()
    a = client.post("/api/crop/analyze", json={"points": r["points"], "threshold": 0.5, "smooth_days": 0}).json()
    assert [p["ndvi"] for p in a["raw"]] == [0.5, 0.3, 0.7]  # same-day readings averaged
    assert [s["date"] for s in a["stressed"]] == ["2026-01-09"]
    p = client.post("/api/crop/image", json={"file_id": tif["id"], "satellite": "Sentinel-2", "date": "2026-02-01"}).json()
    assert 0 < p["ndvi"] < 1

    assert client.delete(f"/api/files/{img['id']}").json()["ok"]
    assert client.post("/api/ndvi", json={"file_id": img["id"]}).status_code == 404
    assert client.get("/").status_code == 200 and b"GeoAI" in client.get("/").content


def test_georeference_recovers_known_transform():
    from pyproj import Transformer
    from rasterio.transform import from_origin as fo
    true = fo(780000, 1440000, 10, 10)  # what s2_tif() uses, EPSG:32643
    to_ll = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    pixels = [(0, 0), (39, 0), (0, 39), (39, 39), (20, 10)]
    gcps = []
    for col, row in pixels:
        lon, lat = to_ll.transform(*(true @ (col, row)))
        gcps.append({"col": col, "row": row, "lon": lon, "lat": lat})

    f = upload("plain.png", png())  # no georeferencing at all
    assert client.post("/api/georef", json={"file_id": f["id"], "gcps": gcps[:2]}).status_code == 400  # < 3 points
    line = [{"col": i, "row": i, "lon": 77 + i / 1e3, "lat": 13 + i / 1e3} for i in range(4)]
    assert "line" in client.post("/api/georef", json={"file_id": f["id"], "gcps": line}).json()["detail"]

    r = client.post("/api/georef", json={"file_id": f["id"], "gcps": gcps, "save": True}).json()
    assert r["crs"] == "EPSG:32643" and r["rmse_m"] < 0.01 and abs(r["pixel_size_m"] - 10) < 1e-6
    (s, w), (n, e) = r["preview"]["bounds"]
    assert 77 < w < e < 78 and 12 < s < n < 14
    new = r["file"]
    assert new["name"] == "plain_georef.tif" and new["meta"]["georeferenced"] and new["meta"]["crs"] == "EPSG:32643"
    out = client.get(r["downloads"][0]["url"]).content
    with MemoryFile(out) as mem, mem.open() as src:
        assert src.count == 3 and src.transform.almost_equals(true, precision=1e-6)
