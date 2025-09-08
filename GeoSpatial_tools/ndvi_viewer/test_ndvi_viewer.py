import streamlit as st
import os
from ndvi_viewer.app import ndvi_viewer

# Mock uploaded files
class MockUploadedFile:
    def __init__(self, file_path):
        self.name = os.path.basename(file_path)
        self.file_path = file_path
    
    def read(self):
        with open(self.file_path, 'rb') as f:
            return f.read()

def test_ndvi_viewer():
    st.title("NDVI Viewer Test")
    
    test_file_path = "sample_data/satellite_images/sentinel2_sample.tif"
    if not os.path.exists(test_file_path):
        st.error(f"Test file not found at: {test_file_path}")
        return
    
    uploaded_files = {
        'tif': [MockUploadedFile(test_file_path)],
        'tiff': [],
        'geotiff': [],
        'jpg': [],
        'png': []
    }
    
    # Run the viewer with our test file
    ndvi_viewer(uploaded_files)

if __name__ == "__main__":
    test_ndvi_viewer()