import streamlit as st
import pandas as pd
from .map_utils import create_heatmap, validate_gps_data, create_drawable_map, get_drawn_features
from utils.visualization import display_map

def gps_heatmapper(uploaded_files):
    st.header("🔥 GPS Heatmapper")
    st.markdown("Create heatmaps from uploaded data or by drawing points directly on a map")
    
    # Initialize session state for drawn points and tags
    if 'drawn_points' not in st.session_state:
        st.session_state.drawn_points = pd.DataFrame(columns=['lat', 'lon', 'tag'])
    
    # Data source selection
    data_source = st.radio(
        "Data source:",
        ["Upload CSV", "Draw on Map"],
        horizontal=True,
        index=0
    )

    if data_source == "Upload CSV":
        handle_csv_upload(uploaded_files)
    else:
        handle_map_drawing()

def handle_csv_upload(uploaded_files):
    """Handle CSV file processing with enhanced features"""
    if 'csv' not in uploaded_files:
        st.warning("No CSV files found. Please upload GPS data or switch to drawing mode.")
        return
    
    selected_file = st.selectbox("Select CSV file", [f.name for f in uploaded_files['csv']])
    file = next(f for f in uploaded_files['csv'] if f.name == selected_file)
    
    try:
        df = pd.read_csv(file)
        
        # Auto-detect coordinate columns
        lat_col, lon_col = detect_coordinate_columns(df)
        
        if None in [lat_col, lon_col]:
            st.error("Could not detect coordinate columns")
            col1, col2 = st.columns(2)
            with col1:
                lat_col = st.selectbox("Select latitude column", df.columns)
            with col2:
                lon_col = st.selectbox("Select longitude column", df.columns)
        
        # Validate data
        if not validate_gps_data(df, lat_col, lon_col):
            st.error("Invalid coordinate values detected")
            return
        
        # Heatmap configuration
        st.subheader("Heatmap Configuration")
        col1, col2 = st.columns(2)
        with col1:
            radius = st.slider("Point radius", 5, 50, 15)
            blur = st.slider("Blur intensity", 5, 50, 15)
        with col2:
            # Weight column selection if available
            weight_col = st.selectbox(
                "Intensity weight column (optional)",
                [None] + [col for col in df.columns if col not in [lat_col, lon_col]]
            )
        
        if st.button("Generate Heatmap"):
            with st.spinner("Generating visualization..."):
                m = create_heatmap(
                    df, 
                    lat_col, 
                    lon_col, 
                    radius=radius, 
                    blur=blur,
                    weight_col=weight_col
                )
                display_map(m)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

def handle_map_drawing():
    """Handle interactive map drawing with enhanced features"""
    st.markdown("""
    ### Draw Points on Map
    1. Use the toolbar (top-left) to place markers
    2. Click on markers to view/edit them
    3. Add tags/labels when placing markers
    4. Generate heatmap when ready
    """)
    
    # Drawing configuration
    st.sidebar.subheader("Drawing Settings")
    enable_clustering = st.sidebar.checkbox("Enable marker clustering", True)
    enable_tagging = st.sidebar.checkbox("Enable point tagging", True)
    
    # Create interactive map with enhanced features
    m = create_drawable_map(
        enable_marker_clustering=enable_clustering,
        enable_editing=True,
        enable_tagging=enable_tagging
    )
    
    # Display the map
    display_map(m)
    
    # Process drawn features
    drawn_features = get_drawn_features(m)
    if drawn_features:
        new_points = pd.DataFrame([{
            'lat': f['location'][0],
            'lon': f['location'][1],
            'tag': f.get('tag')
        } for f in drawn_features])
        
        # Update session state if points changed
        if not st.session_state.drawn_points.equals(new_points):
            st.session_state.drawn_points = new_points
    
    # Display captured points
    if not st.session_state.drawn_points.empty:
        st.subheader("Captured Points")
        st.dataframe(st.session_state.drawn_points)
        
        # Heatmap controls
        st.subheader("Heatmap Settings")
        col1, col2 = st.columns(2)
        with col1:
            radius = st.slider("Heat radius", 5, 50, 15, key="draw_radius")
            blur = st.slider("Blur intensity", 5, 50, 15, key="draw_blur")
        
        if st.button("Generate from Drawn Points"):
            with st.spinner("Creating heatmap..."):
                # Use tags as weights if available
                weight_col = None
                if enable_tagging and 'tag' in st.session_state.drawn_points.columns:
                    try:
                        st.session_state.drawn_points['tag_weight'] = (
                            st.session_state.drawn_points['tag'].astype(float)
                        )
                        weight_col = 'tag_weight'
                    except:
                        pass
                
                hm = create_heatmap(
                    st.session_state.drawn_points,
                    'lat',
                    'lon',
                    radius=radius,
                    blur=blur,
                    weight_col=weight_col
                )
                display_map(hm)
                
            # Export options
            st.subheader("Export Data")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Points as CSV",
                    st.session_state.drawn_points.to_csv(index=False),
                    file_name="drawn_points.csv",
                    mime="text/csv"
                )
            with col2:
                # Export as GeoJSON option
                import geopandas as gpd
                from shapely.geometry import Point
                
                gdf = gpd.GeoDataFrame(
                    st.session_state.drawn_points,
                    geometry=[Point(xy) for xy in zip(
                        st.session_state.drawn_points.lon,
                        st.session_state.drawn_points.lat
                    )]
                )
                st.download_button(
                    "Download as GeoJSON",
                    gdf.to_json(),
                    file_name="points.geojson",
                    mime="application/json"
                )
    else:
        st.info("No points drawn yet - use the map tools to add points")

def detect_coordinate_columns(df):
    """Auto-detect latitude/longitude columns with more options"""
    lat_cols = ['lat', 'latitude', 'y', 'ycoord', 'y_coord', 'ylat']
    lon_cols = ['lon', 'longitude', 'x', 'xcoord', 'x_coord', 'xlon']
    
    # Check for exact matches first
    for lat in lat_cols:
        if lat in df.columns:
            for lon in lon_cols:
                if lon in df.columns:
                    return lat, lon
    
    # Check for case-insensitive matches
    lower_cols = [col.lower() for col in df.columns]
    for lat in lat_cols:
        if lat in lower_cols:
            lat_col = df.columns[lower_cols.index(lat)]
            for lon in lon_cols:
                if lon in lower_cols:
                    lon_col = df.columns[lower_cols.index(lon)]
                    return lat_col, lon_col
    
    return None, None