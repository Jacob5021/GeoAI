import folium
from folium.plugins import HeatMap, Draw, MarkerCluster
import streamlit as st
from utils.visualization import add_basemaps

def create_heatmap(df, lat_col, lon_col, radius=15, blur=15, weight_col=None):
    """Generate Folium heatmap from DataFrame with base map selector"""

    if df.empty:
        return folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    m = folium.Map(
        location=[df[lat_col].mean(), df[lon_col].mean()],
        zoom_start=12,
        tiles=None
    )

    add_basemaps(m)

    # Prepare heatmap data
    if weight_col and weight_col in df.columns:
        heat_data = df[[lat_col, lon_col, weight_col]].values.tolist()
    else:
        heat_data = df[[lat_col, lon_col]].values.tolist()

    HeatMap(heat_data, radius=radius, blur=blur).add_to(m)
    return m


def validate_gps_data(df, lat_col, lon_col):
    """Validate coordinate columns"""
    if not all(col in df.columns for col in [lat_col, lon_col]):
        return False
    try:
        return (df[lat_col].between(-90, 90).all() and
                df[lon_col].between(-180, 180).all())
    except Exception:
        return False


def create_drawable_map(enable_marker_clustering=True, enable_editing=True):
    """Create map with drawing tools"""
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles=None)

    if enable_marker_clustering:
        MarkerCluster().add_to(m)

    draw_options = {
        'polyline': False,
        'polygon': False,
        'circle': False,
        'marker': True,
        'circlemarker': False,
        'rectangle': False
    }

    Draw(
        export=True,
        position='topleft',
        draw_options=draw_options,
        edit_options={"edit": enable_editing}
    ).add_to(m)

    add_basemaps(m)
    return m


def get_drawn_features(map_data):
    """Extract drawn points from Folium map data dictionary"""
    features = []
    if map_data:
        for f in map_data.get("all_drawings") or []:  # None until something is drawn
            if f["geometry"]["type"] == "Point":
                coords = f["geometry"]["coordinates"]
                features.append({
                    "lat": coords[1],
                    "lon": coords[0],
                    "weight": 1  # default numeric weight
                })
    return features
