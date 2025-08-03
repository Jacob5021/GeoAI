import streamlit as st
import os
from utils.geospatial_utils import validate_file

def data_uploader():
    """Drag-and-drop interface for uploading various geospatial files"""
    st.header("📤 Data Uploader")
    st.write("Upload your geospatial data files for analysis across all tools")
    
    uploaded_files = {}
    
    # Drag and drop area
    with st.expander("Upload Files", expanded=True):
        files = st.file_uploader(
            "Drag and drop files here",
            type=["tif", "tiff", "geotiff", "jpg", "png", "csv", "geojson", "shp", "kml"],
            accept_multiple_files=True,
            help="Supported formats: GeoTIFF, JPG, PNG, CSV, GeoJSON, Shapefile"
        )
        
        if files:
            for file in files:
                # Validate file
                if validate_file(file):
                    file_type = os.path.splitext(file.name)[1][1:].lower()
                    if file_type not in uploaded_files:
                        uploaded_files[file_type] = []
                    uploaded_files[file_type].append(file)
                    
                    st.success(f"✅ {file.name} uploaded successfully as {file_type}")
                else:
                    st.error(f"❌ Unsupported file format: {file.name}")
    
    # Display uploaded files by category
    if uploaded_files:
        st.subheader("Uploaded Files")
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown("**Satellite Imagery**")
            for ext in ["tif", "tiff", "geotiff", "jpg", "png"]:
                if ext in uploaded_files:
                    for file in uploaded_files[ext]:
                        st.write(f"- {file.name}")
        
        with cols[1]:
            st.markdown("**Vector Data**")
            for ext in ["geojson", "shp", "kml"]:
                if ext in uploaded_files:
                    for file in uploaded_files[ext]:
                        st.write(f"- {file.name}")
        
        with cols[2]:
            st.markdown("**Tabular Data**")
            if "csv" in uploaded_files:
                for file in uploaded_files["csv"]:
                    st.write(f"- {file.name}")
    
    return uploaded_files