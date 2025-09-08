import streamlit as st
import os
import pandas as pd
import rasterio

def preview_file_metadata(file, file_type):
    """Quick metadata/preview for supported files"""
    try:
        if file_type in ["tif", "tiff", "geotiff"]:
            with rasterio.open(file) as src:
                return {
                    "Driver": src.driver,
                    "CRS": str(src.crs),
                    "Width": src.width,
                    "Height": src.height,
                    "Bands": src.count,
                    "Bounds": str(src.bounds)
                }
        elif file_type == "csv":
            df = pd.read_csv(file, nrows=5)
            return {
                "Columns": list(df.columns),
                "Preview": df.head().to_dict(orient="records")
            }
        elif file_type in ["jpg", "jpeg", "png"]:
            return {"Type": "Image", "Name": file.name}
    except Exception as e:
        return {"Error": str(e)}
    return {}

def data_uploader():
    """Drag-and-drop interface for uploading various geospatial files"""
    st.header("📤 Data Uploader")
    st.write("Upload your geospatial data files for analysis across all tools")

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}

    uploaded_files = {}
    allowed_extensions = ["tif", "tiff", "geotiff", "jpg", "jpeg", "png", "csv", "geojson", "shp", "kml"]

    # Drag and drop area
    with st.expander("Upload Files", expanded=True):
        files = st.file_uploader(
            "Drag and drop files here",
            type=allowed_extensions,
            accept_multiple_files=True,
            help="Supported formats: GeoTIFF, JPG, JPEG, PNG, CSV, GeoJSON, Shapefile, KML"
        )

        if files:
            for file in files:
                file_type = os.path.splitext(file.name)[1][1:].lower()
                if file_type in allowed_extensions:
                    if file_type not in uploaded_files:
                        uploaded_files[file_type] = []
                    uploaded_files[file_type].append(file)

                    st.success(f"✅ {file.name} uploaded successfully as {file_type}")

                    # --- Metadata preview ---
                    with st.expander(f"ℹ️ Metadata: {file.name}"):
                        meta = preview_file_metadata(file, file_type)
                        st.json(meta)
                else:
                    st.error(f"❌ Unsupported file format: {file.name}")

    # Save globally
    if uploaded_files:
        st.session_state.uploaded_files.update(uploaded_files)

        st.subheader("Uploaded Files")
        cols = st.columns(3)

        with cols[0]:
            st.markdown("**🛰️ Satellite Imagery**")
            for ext in ["tif", "tiff", "geotiff", "jpg", "jpeg", "png"]:
                for file in st.session_state.uploaded_files.get(ext, []):
                    st.write(f"- {file.name}")

        with cols[1]:
            st.markdown("**📍 Vector Data**")
            for ext in ["geojson", "shp", "kml"]:
                for file in st.session_state.uploaded_files.get(ext, []):
                    st.write(f"- {file.name}")

        with cols[2]:
            st.markdown("**📑 Tabular Data**")
            for file in st.session_state.uploaded_files.get("csv", []):
                st.write(f"- {file.name}")

    return st.session_state.uploaded_files
