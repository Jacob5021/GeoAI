import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ndvi_viewer.ndvi_processor import (
    calculate_ndvi_from_file,
    get_satellite_bands,
    SATELLITE_PROFILES
)

def crop_monitor(uploaded_files):
    st.header("🌾 Smart Crop Monitoring")
    st.markdown("""
    Monitor crop health through:
    - NDVI time series analysis **or**
    - Direct NDVI calculation from satellite images
    """)
    
    # Initialize session state for time series data
    if 'ndvi_data' not in st.session_state:
        st.session_state.ndvi_data = pd.DataFrame(columns=['date', 'ndvi'])
    
    # Mode selection
    analysis_mode = st.radio(
        "Data Source",
        ["Upload NDVI CSV", "Calculate from Images"],
        horizontal=True
    )
    
    if analysis_mode == "Upload NDVI CSV":
        process_csv_data(uploaded_files)
    else:
        process_image_data(uploaded_files)
    
    # Always show the time series visualization if data exists
    if not st.session_state.ndvi_data.empty:
        visualize_time_series()

def process_csv_data(uploaded_files):
    """Process uploaded CSV time series"""
    if 'csv' not in uploaded_files:
        st.warning("No CSV files found. Switch to image mode or upload files.")
        return
    
    selected_file = st.selectbox("Select CSV file", [f.name for f in uploaded_files['csv']])
    file = next(f for f in uploaded_files['csv'] if f.name == selected_file)
    
    try:
        df = pd.read_csv(file)
        
        # Auto-detect columns with improved logic
        date_col = detect_column(df, ['date','time','timestamp','datetime'], "date")
        ndvi_col = detect_column(df, ['ndvi','vegetation','index','value'], "NDVI")
        
        # Validate data
        df[date_col] = pd.to_datetime(df[date_col])
        if not df[ndvi_col].between(0, 1).all():
            st.error("NDVI values must be between 0 and 1")
            return
        
        # Store in session state
        st.session_state.ndvi_data = df[[date_col, ndvi_col]].rename(
            columns={date_col: 'date', ndvi_col: 'ndvi'}
        ).sort_values('date')
        
        st.success(f"Loaded {len(df)} data points")
        
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")

def process_image_data(uploaded_files):
    """Process uploaded satellite images to calculate NDVI"""
    image_files = []
    for ext in ['tif','tiff','geotiff','jpg','png']:
        image_files.extend(uploaded_files.get(ext, []))
    
    if not image_files:
        st.warning("No image files found. Please upload satellite images.")
        return
    
    selected_file = st.selectbox("Select image", [f.name for f in image_files])
    file = next(f for f in image_files if f.name == selected_file)
    
    # Satellite selection
    satellite = st.selectbox(
        "Satellite/Sensor",
        options=list(SATELLITE_PROFILES.keys()),
        index=0
    )
    
    # Get band defaults but allow override
    bands = get_satellite_bands(satellite)
    col1, col2 = st.columns(2)
    with col1:
        red_band = st.number_input(
            "Red band index",
            min_value=1,
            value=bands['red'],
            help=f"Default for {satellite}: Band {bands['red']}"
        )
    with col2:
        nir_band = st.number_input(
            "NIR band index",
            min_value=1,
            value=bands['nir'],
            help=f"Default for {satellite}: Band {bands['nir']}"
        )
    
    # Date selection for the image
    img_date = st.date_input("Image acquisition date")
    
    if st.button("Calculate NDVI"):
        try:
            with st.spinner("Processing image..."):
                # Calculate mean NDVI for the entire image
                ndvi_value = calculate_ndvi_from_file(file, red_band, nir_band)
                mean_ndvi = np.nanmean(ndvi_value)
                
                # Add to time series
                new_entry = pd.DataFrame({
                    'date': [pd.to_datetime(img_date)],
                    'ndvi': [mean_ndvi]
                })
                
                st.session_state.ndvi_data = pd.concat([
                    st.session_state.ndvi_data,
                    new_entry
                ]).sort_values('date').drop_duplicates('date')
                
                st.success(f"NDVI calculated: {mean_ndvi:.3f}")
                
        except Exception as e:
            st.error(f"NDVI calculation failed: {str(e)}")

def visualize_time_series():
    """Visualize the accumulated NDVI data"""
    st.subheader("NDVI Time Series Analysis")
    df = st.session_state.ndvi_data.sort_values('date')
    
    # Analysis parameters
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider(
            "Stress threshold",
            0.0, 1.0, 0.5, 0.01,
            help="NDVI values below this indicate stress"
        )
    with col2:
        smooth_days = st.slider(
            "Smoothing window (days)",
            0, 30, 7,
            help="Moving average to reduce noise"
        )
    
    # Apply smoothing if requested
    if smooth_days > 0:
        df['ndvi_smooth'] = df['ndvi'].rolling(
            window=f'{smooth_days}D',
            on='date'
        ).mean()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['date'], df['ndvi'], 'o-', label='Raw NDVI', alpha=0.5)
    
    if smooth_days > 0:
        ax.plot(df['date'], df['ndvi_smooth'], '-', label='Smoothed', linewidth=2)
    
    ax.axhline(threshold, color='r', linestyle='--', label='Stress Threshold')
    ax.set_ylim(0, 1)
    ax.set_xlabel("Date")
    ax.set_ylabel("NDVI Value")
    ax.set_title("Crop Health Time Series")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Stress detection
    stressed = df[df['ndvi'] < threshold]
    if not stressed.empty:
        st.warning(f"⚠️ Stress detected on {len(stressed)} dates")
        st.dataframe(stressed)
    else:
        st.success("✅ No stress detected in current data")
    
    # Data export
    with st.expander("💾 Export Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Time Series",
            data=csv,
            file_name="ndvi_time_series.csv",
            mime="text/csv"
        )

def detect_column(df, possible_names, col_type):
    """Helper to detect columns with fallback"""
    col = next((c for c in possible_names if c in df.columns), None)
    if col is None:
        st.error(f"Could not detect {col_type} column")
        return st.selectbox(f"Select {col_type} column", df.columns)
    return col