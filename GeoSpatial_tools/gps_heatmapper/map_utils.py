import folium
from folium.plugins import HeatMap, Draw, MarkerCluster
import pandas as pd

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
    """
    Create map with enhanced drawing tools
    
    Parameters:
    - enable_marker_clustering: Cluster markers when too many are present
    - enable_editing: Allow editing/removing existing markers
    - enable_tagging: Allow adding tags/labels to markers
    """
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
        'edit': enable_editing  # Enable editing of existing markers
    }
    
    # Add drawing tools
    draw_control = Draw(
        export=True,
        position='topleft',
        draw_options=draw_options
    )
    
    if enable_tagging:
        # Add custom HTML for tagging
        m.get_root().html.add_child(folium.Element("""
        <script>
            function addTagToMarker(marker) {
                var tag = prompt("Enter tag/label for this point:");
                if (tag !== null) {
                    marker.bindPopup("Tag: " + tag).openPopup();
                    marker.tag = tag;  // Store tag in marker object
                }
            }
        </script>
        """))
        
        # Add callback for tagging
        m.get_root().html.add_child(folium.Element("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                var map = {map_variable_name};
                map.on('draw:created', function(e) {
                    var layer = e.layer;
                    if (e.layerType === 'marker') {
                        addTagToMarker(layer);
                    }
                    {cluster_variable_name}.addLayer(layer);
                });
            });
        </script>
        """.replace('{map_variable_name}', m.get_name())
         .replace('{cluster_variable_name}', 'marker_cluster' if enable_marker_clustering else 'map')))
    
    draw_control.add_to(m)
    
    # Add base map selector
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('Stamen Terrain', name='Stamen Terrain').add_to(m)
    folium.TileLayer('Stamen Toner', name='Stamen Toner').add_to(m)
    folium.TileLayer('CartoDB Positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB Dark_Matter', name='CartoDB Dark_Matter').add_to(m)
    folium.LayerControl().add_to(m)
    
    return m

def get_drawn_features(map_object):
    """Get all drawn features from map with their tags"""
    features = []
    for child in map_object._children.values():
        if isinstance(child, folium.FeatureGroup):
            for marker in child._children.values():
                if isinstance(marker, folium.Marker):
                    feature = {
                        'location': marker.location,
                        'tag': getattr(marker, 'tag', None)
                    }
                    features.append(feature)
    return features