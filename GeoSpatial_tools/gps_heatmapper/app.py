import streamlit as st
import pandas as pd
from .map_utils import create_heatmap, validate_gps_data, create_drawable_map, get_drawn_features
from utils.visualization import display_map
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point


def gps_heatmapper(uploaded_files):
    st.header("🔥 GPS Heatmapper")
    st.markdown("Create heatmaps from uploaded CSV or by drawing points directly on a map")

    if 'drawn_points' not in st.session_state:
        st.session_state.drawn_points = pd.DataFrame(columns=['lat', 'lon', 'weight'])

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
    if 'csv' not in uploaded_files:
        st.warning("No CSV files uploaded")
        return

    selected_file = st.selectbox("Select CSV file", [f.name for f in uploaded_files['csv']])
    file = next(f for f in uploaded_files['csv'] if f.name == selected_file)

    try:
        df = pd.read_csv(file)

        lat_col, lon_col = detect_coordinate_columns(df)
        if None in [lat_col, lon_col]:
            st.error("Could not auto-detect coordinates")
            col1, col2 = st.columns(2)
            with col1:
                lat_col = st.selectbox("Latitude column", df.columns)
            with col2:
                lon_col = st.selectbox("Longitude column", df.columns)

        if not validate_gps_data(df, lat_col, lon_col):
            st.error("Invalid coordinates detected")
            return

        st.subheader("Heatmap Settings")
        col1, col2 = st.columns(2)
        with col1:
            radius = st.slider("Point radius", 5, 50, 15)
            blur = st.slider("Blur intensity", 5, 50, 15)
        with col2:
            weight_col = st.selectbox(
                "Intensity weight column (optional)",
                [None] + [c for c in df.columns if c not in [lat_col, lon_col]]
            )

        if st.button("Generate Heatmap"):
            hm = create_heatmap(df, lat_col, lon_col, radius, blur, weight_col)
            display_map(hm)

    except Exception as e:
        st.error(f"Error: {str(e)}")


def handle_map_drawing():
    st.markdown("""
    ### Draw Points
    1. Use the toolbar to place markers
    2. Edit weights in the table below
    3. Generate heatmap when ready
    """)

    enable_clustering = st.sidebar.checkbox("Enable marker clustering", True)
    m = create_drawable_map(enable_marker_clustering=enable_clustering)

    map_data = st_folium(m, width=700, height=500)
    drawn_points = get_drawn_features(map_data)

    if drawn_points:
        st.session_state.drawn_points = pd.DataFrame(drawn_points)

    if not st.session_state.drawn_points.empty:
        st.subheader("Points & Weights")
        edited_points = st.data_editor(
            st.session_state.drawn_points,
            num_rows="dynamic",
            use_container_width=True
        )
        st.session_state.drawn_points = edited_points

        st.subheader("Heatmap Settings")
        col1, col2 = st.columns(2)
        with col1:
            radius = st.slider("Heat radius", 5, 50, 15, key="draw_radius")
        with col2:
            blur = st.slider("Blur intensity", 5, 50, 15, key="draw_blur")

        if st.button("Generate Heatmap"):
            hm = create_heatmap(st.session_state.drawn_points,
                                'lat', 'lon',
                                radius=radius,
                                blur=blur,
                                weight_col='weight')
            st_folium(hm, width=700, height=500)

        st.subheader("Export Data")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download CSV",
                st.session_state.drawn_points.to_csv(index=False),
                file_name="points.csv",
                mime="text/csv"
            )
        with col2:
            gdf = gpd.GeoDataFrame(
                st.session_state.drawn_points,
                geometry=[Point(xy) for xy in zip(
                    st.session_state.drawn_points.lon,
                    st.session_state.drawn_points.lat
                )]
            )
            st.download_button(
                "Download GeoJSON",
                gdf.to_json(),
                file_name="points.geojson",
                mime="application/json"
            )
    else:
        st.info("No points drawn yet")


def detect_coordinate_columns(df):
    lat_cols = ['lat', 'latitude', 'y', 'ycoord', 'y_coord', 'ylat']
    lon_cols = ['lon', 'longitude', 'x', 'xcoord', 'x_coord', 'xlon']

    for lat in lat_cols:
        if lat in df.columns:
            for lon in lon_cols:
                if lon in df.columns:
                    return lat, lon

    lower_cols = [c.lower() for c in df.columns]
    for lat in lat_cols:
        if lat in lower_cols:
            lat_col = df.columns[lower_cols.index(lat)]
            for lon in lon_cols:
                if lon in lower_cols:
                    lon_col = df.columns[lower_cols.index(lon)]
                    return lat_col, lon_col

    return None, None
