# Geospatial AI Tools Suite

A comprehensive toolkit for analyzing satellite and geospatial data with AI capabilities.

## Features

- **Data Uploader**: Drag-and-drop interface for various geospatial file formats
- **NDVI Viewer**: Visualize and analyze vegetation indices
- **Land Use Classifier**: Classify satellite imagery into urban, forest, water, etc.
- **GPS Heatmapper**: Create interactive heatmaps from GPS data
- **Pollution Visualizer**: Analyze NO₂ and other pollution data
- **Crop Monitoring**: Track vegetation health over time
- **Satellite Object Detection**: Detect objects in satellite imagery using YOLO

## Installation

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `streamlit run main_dashboard.py`

## Usage

1. Start by uploading your data using the Data Uploader
2. Navigate to the appropriate tool for your analysis
3. All tools will automatically detect relevant files from your uploads
4. Adjust parameters as needed and view results