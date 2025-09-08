import folium
from folium.plugins import HeatMap, Draw, MarkerCluster
import pandas as pd
import streamlit as st

def create_heatmap(df, lat_col, lon_col, radius=15, blur=15, weight_col=None):
    """Generate Folium heatmap from DataFrame with base map selector"""

    # Base map options
    base_maps = {
        'OpenStreetMap': 'OpenStreetMap',
        'Stamen Terrain': 'Stamen Terrain',
        'Stamen Toner': 'Stamen Toner',
        'CartoDB Positron': 'CartoDB Positron',
        'CartoDB Dark_Matter': 'CartoDB Dark_Matter'
    }

    m = folium.Map(
        location=[df[lat_col].mean(), df[lon_col].mean()],
        zoom_start=12,
        tiles=base_maps['OpenStreetMap']  # Default base map
    )

    # Add base map selector
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('Stamen Terrain', name='Stamen Terrain').add_to(m)
    folium.TileLayer('Stamen Toner', name='Stamen Toner').add_to(m)
    folium.TileLayer('CartoDB Positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB Dark_Matter', name='CartoDB Dark_Matter').add_to(m)
    folium.LayerControl().add_to(m)

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
    except:
        return False

def create_drawable_map(enable_marker_clustering=True, enable_editing=True, enable_tagging=False):
    """Create map with enhanced drawing tools"""

    m = folium.Map(location=[20, 0], zoom_start=2)

    # Initialize marker cluster if enabled
    if enable_marker_clustering:
        marker_cluster = MarkerCluster().add_to(m)

    # Drawing tools configuration
    draw_options = {
        'polyline': False,
        'polygon': False, 
        'circle': False,
        'marker': True,
        'circlemarker': False,
        'rectangle': False,
        'edit': enable_editing
    }

    # Add drawing tools
    draw_control = Draw(
        export=True,
        position='topleft',
        draw_options=draw_options
    )

    draw_control.add_to(m)

    if enable_tagging:
        # Add custom JavaScript for tagging functionality
        tag_script = """
        <script>
        var taggedFeatures = {};

        map.on('draw:created', function(e) {
            var layer = e.layer;
            var feature_id = layer._leaflet_id;

            var tag = prompt("Enter a tag/label for this point (optional):");
            if (tag) {
                taggedFeatures[feature_id] = tag;
                layer.bindPopup("Tag: " + tag).openPopup();
            }
        });
        </script>
        """

        m.get_root().html.add_child(folium.Element(tag_script))

    return m

def get_drawn_features(folium_map):
    """Extract drawn features from Folium map - placeholder implementation"""
    # This would need streamlit-folium integration for real functionality
    features = []
    return features
