# Geospatial AI Tools Suite

A comprehensive toolkit for analyzing satellite and geospatial data with AI capabilities.

## Features

- **Data library**: Drag-and-drop upload for GeoTIFF, imagery, CSV and vector files
- **NDVI Viewer**: Visualize and analyze vegetation indices
- **Land Use Classifier**: Classify satellite imagery into urban, forest, water, etc.
- **GPS Heatmapper**: Create interactive heatmaps from GPS data
- **Pollution Visualizer**: Analyze NO₂ and other pollution data
- **Crop Monitoring**: Track vegetation health over time
- **Satellite Object Detection**: Detect objects in satellite imagery using YOLO
- **Georeference**: Pin an image to the map with ground control points and export a georeferenced GeoTIFF

## Quick start (any machine)

Needs Python 3.10+ and git.

```
git clone https://github.com/Jacob5021/GeoAI.git
cd GeoAI/GeoSpatial_tools
python run.py
```

Open http://localhost:8599 and create an account. The first run creates `venv/` and installs the
requirements (PyTorch is a large download); later runs start in seconds.

## Where data is stored

Each installation keeps its **own local database**, created automatically on first start:

```
GeoSpatial_tools/data/
  geoai.db                         users, sessions, datasets, results (SQLite)
  <username>/
    <dataset>_<id>/
      <original file>
      results/<tool>_<date>_<id>/  outputs (GeoTIFF/PNG/CSV) + result.json
```

- `data/` is in `.gitignore`: pulling the repo on another system starts with an empty database there,
  and nothing from it is ever pushed.
- Results are saved automatically next to their dataset. Running a tool again with the same settings
  reopens the saved result instead of recomputing (use **Re-run** to force it).
- Back up or move an installation by copying `data/` (with the server stopped).
- Put the data elsewhere with `GEOAI_DATA_DIR=/path/to/data python run.py`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEOAI_HOST` | `127.0.0.1` | Listen address. `0.0.0.0` exposes it to your network. |
| `GEOAI_PORT` | `8599` | Port |
| `GEOAI_DATA_DIR` | `./data` | Database and file storage |
| `GEOAI_ALLOW_SIGNUP` | `true` | Set `false` once your users have accounts |
| `GEOAI_MAX_UPLOAD_MB` | `1024` | Per-file upload limit |
| `GEOAI_SECURE_COOKIES` | `false` | Set `true` when served over HTTPS |

## Development

- Manual setup instead of `run.py`: `python -m venv venv`, activate it, `pip install -r requirements.txt`,
  then `uvicorn server:app --port 8599`.
- Tests: `pytest test_core.py test_api.py` (they use a temporary data folder, never `data/`).
- The backend is FastAPI (`server.py`, storage in `storage.py`); the frontend is plain HTML/CSS/JS in
  `frontend/` with no build step.

Models: object detection uses the bundled COCO `yolov8n.pt`. The DeepLab land-use option
appears only if `landuse_classifier/deeplabv3_finetuned_RS_openearthmap_v2.pth` exists.

## Security

Accounts are separate: each user sees only their own datasets and results. Passwords are hashed with
scrypt; sessions are HttpOnly cookies; repeated failed sign-ins are throttled. The server listens on
localhost by default. Before exposing it (`GEOAI_HOST=0.0.0.0`), consider disabling sign-up and
serving it behind HTTPS.
