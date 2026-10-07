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

## Installation

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `uvicorn server:app --port 8599`, then open http://localhost:8599
6. Run the tests: `pytest test_core.py test_api.py`

The backend is FastAPI (`server.py`); the frontend is plain HTML/CSS/JS in `frontend/` with no build step.
Uploaded files are kept in server memory and cleared on restart.

Models: object detection uses the bundled COCO `yolov8n.pt`. The DeepLab land-use option
appears only if `landuse_classifier/deeplabv3_finetuned_RS_openearthmap_v2.pth` exists.

## Security

The server has no login. By default `uvicorn` listens on localhost only; don't pass
`--host 0.0.0.0` on an untrusted network. Uploads are capped at 1 GB
(`GEOAI_MAX_UPLOAD_MB` to change).
