import streamlit as st
import pandas as pd
import numpy as np
import folium
import rasterio
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from utils.geospatial_utils import validate_pollution_data
import matplotlib.pyplot as plt
from io import BytesIO
import geopandas as gpd
from folium.features import GeoJsonTooltip

def pollution_visualizer(uploaded_files):
    st.header("🌫️ Pollution Visualizer")
    st.markdown("""
    Visualize atmospheric pollution data from:
    - CSV files (point measurements)
    - GeoTIFF files (raster data from satellites)
    """)
    
    # File selection with type filtering
    available_files = []
    for ext in ['csv', 'tif', 'tiff', 'geotiff']:
        available_files.extend(uploaded_files.get(ext, []))
    
    if not available_files:
        st.warning("Please upload CSV or GeoTIFF files with pollution data")
        return
    
    selected_file = st.selectbox("Select data file", [f.name for f in available_files])
    file = next(f for f in available_files if f.name == selected_file)
    
    # File type handling
    if file.name.lower().endswith('.csv'):
        process_csv_file(file)
    else:
        process_geotiff_file(file)

def process_csv_file(file):
    """Handle CSV pollution data"""
    try:
        df = pd.read_csv(file)
        
        # Auto-detect columns with improved logic
        lat_col, lon_col, pollution_col = detect_pollution_columns(df)
        
        # Detect all potential pollution columns
        pollution_columns = [c for c in df.columns if c.lower() in 
                           ['no2', 'pm25', 'pm10', 'co', 'so2', 'o3', 
                            'value', 'concentration', 'aqi', 'pm2_5']]
        
        if None in [lat_col, lon_col]:
            st.error("Latitude/Longitude columns not detected")
            col1, col2 = st.columns(2)
            with col1:
                lat_col = st.selectbox("Latitude column", df.columns)
                lon_col = st.selectbox("Longitude column", df.columns)
        
        # If multiple pollution columns found, let user select
        if len(pollution_columns) > 1:
            pollution_col = st.selectbox("Select pollutant to visualize", pollution_columns)
        elif pollution_col is None:
            pollution_col = st.selectbox("Pollution column", 
                                       [c for c in df.columns if c not in [lat_col, lon_col]])
        
        # Add threshold filter
        min_val, max_val = float(df[pollution_col].min()), float(df[pollution_col].max())
        threshold = st.slider(
            "Filter by concentration range",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )
        df = df[(df[pollution_col] >= threshold[0]) & (df[pollution_col] <= threshold[1])]
        
        # Validate data
        validation = validate_pollution_data(df, lat_col, lon_col, pollution_col)
        if not validation['is_valid']:
            st.error(f"Data validation failed: {validation['message']}")
            return
        
        # Visualization
        st.subheader(f"{pollution_col} Concentration")
        
        # Normalize data
        df['normalized'] = (df[pollution_col] - df[pollution_col].min()) / \
                          (df[pollution_col].max() - df[pollution_col].min() + 1e-10)
        
        # Create map
        m = folium.Map(
            location=[df[lat_col].mean(), df[lon_col].mean()],
            zoom_start=10,
            tiles="CartoDB dark_matter"
        )
        
        # Add heatmap
        HeatMap(
            df[[lat_col, lon_col, 'normalized']].values,
            radius=15,
            blur=20,
            gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(m)
        
        # Add markers with popups
        for idx, row in df.iterrows():
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=3,
                color='white',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup=f"{pollution_col}: {row[pollution_col]:.2f}"
            ).add_to(m)
        
        # Add shapefile overlay option
        with st.expander("Add Boundary Overlay"):
            boundary_file = st.file_uploader("Upload boundary file (GeoJSON/Shapefile)", 
                                           type=['geojson', 'shp', 'zip'])
            if boundary_file:
                try:
                    if boundary_file.name.endswith('.geojson'):
                        gdf = gpd.read_file(boundary_file)
                    else:  # Shapefile
                        gdf = gpd.read_file(boundary_file)
                    
                    folium.GeoJson(
                        gdf,
                        style_function=lambda x: {
                            'color': 'white',
                            'weight': 2,
                            'fillOpacity': 0
                        },
                        tooltip=GeoJsonTooltip(
                            fields=list(gdf.columns),
                            aliases=[f.capitalize() for f in gdf.columns],
                            localize=True
                        )
                    ).add_to(m)
                    st.success("Boundary overlay added successfully")
                except Exception as e:
                    st.error(f"Failed to load boundary file: {str(e)}")
        
        folium_static(m, width=700, height=500)
        
        # Statistics
        show_pollution_stats(df, pollution_col)
        
    except Exception as e:
        st.error(f"CSV processing error: {str(e)}")

def process_geotiff_file(file):
    """Handle GeoTIFF pollution data with pollutant detection"""
    try:
        with st.spinner("Processing satellite data..."):
            with rasterio.open(file) as src:
                # Read first band (pollution data)
                data = src.read(1)
                bounds = src.bounds
                
                # Try to detect pollutant from metadata
                pollutant_name = None
                if src.descriptions and src.descriptions[0]:
                    pollutant_name = src.descriptions[0]  # band description
                elif src.tags() and 'pollutant' in src.tags():
                    pollutant_name = src.tags()['pollutant']
                
                # If not detected, ask user
                pollutant_name = st.text_input(
                    "Enter pollutant type for this GeoTIFF (e.g., PM2.5, NO2)",
                    value=pollutant_name or ""
                )
                
                if not pollutant_name:
                    st.warning("Pollutant type not specified. Visualization will proceed without label.")
                
                # Create normalized version for visualization
                norm_data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-10)
                
                # Threshold filtering
                min_val, max_val = float(np.nanmin(data)), float(np.nanmax(data))
                threshold = st.slider(
                    f"Filter by concentration range ({pollutant_name})",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
                masked_data = np.where((data >= threshold[0]) & (data <= threshold[1]), data, np.nan)
                
                # Plot raster
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(masked_data, cmap='inferno')
                plt.colorbar(im, ax=ax, label=f'{pollutant_name} Concentration')
                ax.set_title(f"{pollutant_name} Concentration Raster" if pollutant_name else "Pollution Concentration Raster")
                ax.axis('off')
                st.pyplot(fig)
                
                # Interactive map
                st.subheader("Interactive Map")
                m = folium.Map(
                    location=[(bounds.top + bounds.bottom)/2, (bounds.left + bounds.right)/2],
                    zoom_start=8
                )
                
                folium.raster_layers.ImageOverlay(
                    image=norm_data.astype(np.float32),
                    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                    colormap=lambda x: (1, 0, 0, x),  # Red gradient
                    opacity=0.7
                ).add_to(m)
                
                # Add shapefile overlay option
                with st.expander("Add Boundary Overlay"):
                    boundary_file = st.file_uploader("Upload boundary file (GeoJSON/Shapefile/ZIP)", 
                                                    type=['geojson', 'shp', 'zip'])
                    if boundary_file:
                        try:
                            import zipfile, tempfile
                            if boundary_file.name.endswith('.zip'):
                                with tempfile.TemporaryDirectory() as tmpdir:
                                    zf = zipfile.ZipFile(boundary_file)
                                    zf.extractall(tmpdir)
                                    gdf = gpd.read_file(tmpdir)
                            else:
                                gdf = gpd.read_file(boundary_file)
                            
                            folium.GeoJson(
                                gdf,
                                style_function=lambda x: {'color': 'white','weight': 2,'fillOpacity': 0},
                                tooltip=GeoJsonTooltip(
                                    fields=list(gdf.columns),
                                    aliases=[f.capitalize() for f in gdf.columns],
                                    localize=True
                                )
                            ).add_to(m)
                            st.success("Boundary overlay added successfully")
                        except Exception as e:
                            st.error(f"Failed to load boundary file: {str(e)}")
                
                folium_static(m)
                
                # Download options
                with st.expander("Download Options"):
                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    st.download_button(
                        "Download Visualization",
                        buf.getvalue(),
                        file_name=f"{pollutant_name or 'pollution_map'}.png",
                        mime="image/png"
                    )
                    
    except Exception as e:
        st.error(f"GeoTIFF processing error: {str(e)}")


def detect_pollution_columns(df):
    """Enhanced column detection"""
    # Latitude candidates
    lat_col = next((c for c in ['lat', 'latitude', 'y', 'ycoord'] 
                   if c in df.columns), None)
    
    # Longitude candidates
    lon_col = next((c for c in ['lon', 'longitude', 'x', 'xcoord'] 
                   if c in df.columns), None)
    
    # Pollution candidates
    pollution_col = next((c for c in ['no2', 'pm25', 'pm10', 'co', 'so2', 'o3', 
                                    'value', 'concentration', 'aqi', 'pm2_5']
                        if c in df.columns), None)
    
    return lat_col, lon_col, pollution_col

def show_pollution_stats(df, pollution_col):
    """Display pollution statistics"""
    st.subheader("Pollution Statistics")
    
    stats = {
        "Max": df[pollution_col].max(),
        "Min": df[pollution_col].min(),
        "Mean": df[pollution_col].mean(),
        "Std Dev": df[pollution_col].std()
    }
    
    cols = st.columns(4)
    for i, (name, value) in enumerate(stats.items()):
        cols[i].metric(name, f"{value:.2f}")
    
    # Histogram - FIXED: Explicitly pass the figure to st.pyplot()
    fig, ax = plt.subplots()
    ax.hist(df[pollution_col], bins=20, color='red', alpha=0.7)
    ax.set_xlabel("Concentration")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Pollution Values")
    st.pyplot(fig)  # Explicitly pass the figure object