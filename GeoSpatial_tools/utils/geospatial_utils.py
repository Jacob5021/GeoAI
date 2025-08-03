import os
import rasterio
from fiona import drvsupport
from geopandas import read_file
import pandas as pd

def validate_file(file):
    """Validate uploaded geospatial file"""
    valid_extensions = [
        '.tif', '.tiff', '.geotiff', '.jpg', '.png', 
        '.csv', '.geojson', '.shp', '.kml'
    ]
    ext = os.path.splitext(file.name)[1].lower()
    return ext in valid_extensions

def load_geospatial_file(file):
    """Load geospatial file based on its type"""
    ext = os.path.splitext(file.name)[1].lower()
    
    if ext in ['.tif', '.tiff', '.geotiff']:
        return rasterio.open(file)
    elif ext in ['.jpg', '.png']:
        return file  # Handle as image file
    elif ext in ['.geojson', '.shp', '.kml']:
        return read_file(file)
    elif ext == '.csv':
        return pd.read_csv(file)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def get_crs(file):
    """Get CRS of a geospatial file"""
    ext = os.path.splitext(file.name)[1].lower()
    
    if ext in ['.tif', '.tiff', '.geotiff']:
        with rasterio.open(file) as src:
            return src.crs
    elif ext in ['.geojson', '.shp', '.kml']:
        gdf = read_file(file)
        return gdf.crs
    return None

def validate_pollution_data(df, lat_col, lon_col, pollution_col):
    """Comprehensive pollution data validation"""
    errors = []
    
    # Column checks
    if lat_col not in df.columns:
        errors.append(f"Missing latitude column: {lat_col}")
    if lon_col not in df.columns:
        errors.append(f"Missing longitude column: {lon_col}")
    if pollution_col not in df.columns:
        errors.append(f"Missing pollution column: {pollution_col}")
    
    # Data checks
    if not errors:
        try:
            if not pd.api.types.is_numeric_dtype(df[pollution_col]):
                errors.append("Pollution data must be numeric")
                
            if (df[lat_col] < -90).any() or (df[lat_col] > 90).any():
                errors.append("Latitude out of range (-90 to 90)")
                
            if (df[lon_col] < -180).any() or (df[lon_col] > 180).any():
                errors.append("Longitude out of range (-180 to 180)")
        except:
            errors.append("Data validation failed")
    
    return {
        'is_valid': len(errors) == 0,
        'message': "; ".join(errors) if errors else "Data is valid"
    }