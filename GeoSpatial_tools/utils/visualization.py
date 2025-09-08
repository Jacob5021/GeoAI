import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import numpy as np

def show_about():
    """Display about information"""
    st.header("About Geospatial AI Tools")
    st.write("""
    This application provides a suite of tools for analyzing satellite and geospatial data.

    **Features include:**

    - NDVI calculation and visualization
    - Land use classification  
    - GPS heatmap generation
    - Pollution data analysis
    - Crop health monitoring
    - Object detection in satellite imagery

    """)

    st.markdown("""
    ### How to Use

    1. Start by uploading your data in the **Data Uploader** tab
    2. Navigate to the tool you want to use
    3. The tool will automatically detect relevant files from your uploads
    4. Adjust parameters as needed and view results

    """)

def plot_ndvi(ndvi_array, title="NDVI Map"):
    """Plot NDVI array"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ndvi_array, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='NDVI Value')
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)

def create_heatmap(data, latitude_col='lat', longitude_col='lon', zoom=12):
    """Create interactive heatmap"""
    m = folium.Map(
        location=[data[latitude_col].mean(), data[longitude_col].mean()], 
        zoom_start=zoom
    )
    heat_data = [[row[latitude_col], row[longitude_col]] for _, row in data.iterrows()]
    HeatMap(heat_data).add_to(m)
    folium_static(m)

def display_map(folium_map, width=700, height=500):
    """
    Display a Folium map in Streamlit

    Args:
        folium_map: Folium map object
        width: Map width in pixels
        height: Map height in pixels
    """
    try:
        folium_static(folium_map, width=width, height=height)
    except Exception as e:
        st.error(f"Error displaying map: {str(e)}")
        # Fallback: try to display with default settings
        try:
            folium_static(folium_map)
        except:
            st.error("Unable to display map. Please check your data and try again.")

def plot_time_series(df, x_col, y_col, title="Time Series", threshold=None):
    """
    Plot time series data with optional threshold line

    Args:
        df: DataFrame with time series data
        x_col: Column name for x-axis (typically dates)
        y_col: Column name for y-axis (values)
        title: Plot title
        threshold: Optional threshold line to display
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[x_col], df[y_col], 'o-', linewidth=2, markersize=4)

    if threshold is not None:
        ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
        ax.legend()

    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Improve date formatting if x_col contains dates
    if 'date' in x_col.lower() or 'time' in x_col.lower():
        plt.xticks(rotation=45)
        plt.tight_layout()

    st.pyplot(fig)

def create_comparison_plot(data1, data2, labels, title="Comparison"):
    """
    Create side-by-side comparison plots

    Args:
        data1: First dataset
        data2: Second dataset  
        labels: List of labels for the datasets
        title: Overall title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot first dataset
    if len(data1.shape) == 2:  # 2D array (image)
        im1 = ax1.imshow(data1, cmap='viridis')
        plt.colorbar(im1, ax=ax1)
    else:  # 1D array
        ax1.plot(data1)
    ax1.set_title(labels[0])

    # Plot second dataset
    if len(data2.shape) == 2:  # 2D array (image)
        im2 = ax2.imshow(data2, cmap='viridis')
        plt.colorbar(im2, ax=ax2)
    else:  # 1D array
        ax2.plot(data2)
    ax2.set_title(labels[1])

    fig.suptitle(title)
    plt.tight_layout()
    st.pyplot(fig)

def display_statistics_table(df, columns=None):
    """
    Display descriptive statistics in a formatted table

    Args:
        df: DataFrame to analyze
        columns: Specific columns to analyze (if None, uses all numeric columns)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    stats_df = df[columns].describe()
    st.subheader("Descriptive Statistics")
    st.dataframe(stats_df.round(3))

    return stats_df
