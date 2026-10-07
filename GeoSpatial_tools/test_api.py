"""End-to-end API checks: accounts, uploads, every tool endpoint, stored results."""
import glob
import io
import os
import tempfile

os.environ["GEOAI_DATA_DIR"] = tempfile.mkdtemp(prefix="geoai-test-")  # before importing the app

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

import server
import storage
from server import app


def signed_in_client(username, password="correct-horse"):
    c = TestClient(app)
    r = c.post("/api/auth/register", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return c


client = signed_in_client("alice")


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
    assert r["result"]["cached"] is False
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
    assert TestClient(app).get("/").status_code == 200 and b"GeoAI" in TestClient(app).get("/").content


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


def test_accounts_required_and_isolated():
    anon = TestClient(app)
    assert anon.get("/api/files").status_code == 401
    assert anon.post("/api/ndvi", json={"file_id": "x"}).status_code == 401
    assert anon.post("/api/auth/register", json={"username": "a", "password": "longenough"}).status_code == 400
    assert anon.post("/api/auth/register", json={"username": "alice", "password": "longenough"}).json()["detail"] == "That username is taken"
    assert anon.post("/api/auth/login", json={"username": "alice", "password": "wrong-pass"}).status_code == 401

    f = upload("mine.png", png())
    r = client.post("/api/ndvi", json={"file_id": f["id"]}).json()
    bob = signed_in_client("bob")
    assert bob.get("/api/files").json() == []
    assert bob.get(f"/api/files/{f['id']}/thumb").status_code == 404
    assert bob.post("/api/ndvi", json={"file_id": f["id"]}).status_code == 404
    assert bob.get(r["downloads"][0]["url"]).status_code == 404
    assert bob.get(f"/api/results/{r['result']['id']}").status_code == 404
    assert bob.delete(f"/api/files/{f['id']}").status_code == 404

    bob.post("/api/auth/logout")
    assert bob.get("/api/files").status_code == 401


def test_password_change_and_throttle():
    c = signed_in_client("carol", "first-password")
    assert c.post("/api/auth/password", json={"current": "nope-nope", "new": "second-password"}).status_code == 400
    assert c.post("/api/auth/password", json={"current": "first-password", "new": "second-password"}).status_code == 200
    assert c.get("/api/auth/me").json()["username"] == "carol"  # this browser stays signed in
    assert TestClient(app).post("/api/auth/login", json={"username": "carol", "password": "second-password"}).status_code == 200
    t = TestClient(app)
    codes = [t.post("/api/auth/login", json={"username": "carol", "password": "bad-password"}).status_code for _ in range(11)]
    assert codes[:10] == [401] * 10 and codes[10] == 429
    assert storage.hash_password("x") != storage.hash_password("x")  # salted
    assert storage.verify_password("x", storage.hash_password("x")) and not storage.verify_password("y", storage.hash_password("x"))


def test_results_saved_next_to_dataset_and_reused():
    f = upload("Field A.tif", s2_tif())
    first = client.post("/api/ndvi", json={"file_id": f["id"], "satellite": "Sentinel-2"}).json()
    rid = first["result"]["id"]

    again = client.post("/api/ndvi", json={"file_id": f["id"], "satellite": "Sentinel-2"}).json()
    assert again["result"]["cached"] is True and again["result"]["id"] == rid and again["stats"] == first["stats"]
    other = client.post("/api/ndvi", json={"file_id": f["id"], "satellite": "Landsat-8"}).json()  # different settings
    assert other["result"]["id"] != rid and other["result"]["cached"] is False

    rerun = client.post("/api/ndvi", json={"file_id": f["id"], "satellite": "Sentinel-2", "force": True}).json()
    assert rerun["result"]["cached"] is False and rerun["result"]["id"] != rid
    assert client.get(f"/api/results/{rid}").status_code == 404  # replaced, not duplicated

    # Readable folders: data/<user>/<dataset>_<id>/{original, results/ndvi_<time>_<id>/...}
    folder = glob.glob(os.path.join(storage.DATA_DIR, "alice", f"Field_A_{f['id']}"))[0]
    assert os.path.exists(os.path.join(folder, "Field_A.tif"))
    runs = sorted(os.listdir(os.path.join(folder, "results")))
    assert len(runs) == 2 and all(r.startswith("ndvi_") for r in runs)
    assert sorted(os.listdir(os.path.join(folder, "results", runs[0]))) == ["ndvi_Field_A.png", "ndvi_Field_A.tif", "result.json"]

    # Library lists results with their dataset
    listed = next(x for x in client.get("/api/files").json() if x["id"] == f["id"])
    assert {r["summary"]["mode"] for r in listed["results"]} == {"Real NDVI"} and len(listed["results"]) == 2

    # "Restart": drop every in-memory cache; everything comes back from disk + SQLite
    for cache in (server.BYTES_CACHE, server.PREVIEW_CACHE, server.RASTER_CACHE):
        cache.clear()
    fresh = TestClient(app)
    fresh.post("/api/auth/login", json={"username": "alice", "password": "correct-horse"})
    payload = fresh.get(f"/api/results/{rerun['result']['id']}").json()
    assert payload["stats"] == rerun["stats"] and payload["image"] == rerun["image"]
    assert fresh.get(payload["downloads"][1]["url"]).content[:2] in (b"II", b"MM")
    assert fresh.get(f"/api/files/{f['id']}/download").content == s2_tif()

    # Deleting the dataset removes its folder and results
    assert client.delete(f"/api/files/{f['id']}").json()["ok"]
    assert not os.path.exists(folder) and client.get(f"/api/results/{rerun['result']['id']}").status_code == 404


def test_rename_user_moves_data():
    c = signed_in_client("dave")
    f = c.post("/api/files", files=[("files", ("d.png", png()))]).json()[0]
    r = c.post("/api/ndvi", json={"file_id": f["id"]}).json()
    storage.rename_user("dave", "Jacob2")
    assert not os.path.exists(os.path.join(storage.DATA_DIR, "dave"))
    assert c.get("/api/auth/me").json()["username"] == "Jacob2"  # existing session still valid
    assert c.get(f"/api/files/{f['id']}/download").content == png()  # same bytes, read from the new folder
    assert c.get(r["downloads"][0]["url"]).status_code == 200 and c.get(f"/api/results/{r['result']['id']}").status_code == 200
    assert TestClient(app).post("/api/auth/login", json={"username": "dave", "password": "correct-horse"}).status_code == 401
    assert TestClient(app).post("/api/auth/login", json={"username": "Jacob2", "password": "correct-horse"}).status_code == 200
