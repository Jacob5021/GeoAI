import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os
import torch
import torchvision.transforms as T
import pandas as pd
import cv2

# ================== CLASS LABELS & COLORS ==================
LAND_USE_CLASSES = {
    0: "Background",
    1: "Bareland",
    2: "Rangeland",
    3: "Tree",
    4: "Developed land",
    5: "Road",
    6: "Water",
    7: "Agriculture land",
    8: "Building"
}

CLASS_COLORS = {
    0: [0, 0, 0],        # Background - Black
    1: [210, 180, 140],  # Bareland - Tan
    2: [255, 228, 181],  # Rangeland - Moccasin
    3: [34, 139, 34],    # Tree - Forest Green
    4: [128, 128, 128],  # Developed land - Gray
    5: [255, 255, 255],  # Road - White 
    6: [0, 0, 255],      # Water - Blue
    7: [255, 255, 0],    # Agriculture - Yellow
    8: [178, 34, 34]     # Building - Firebrick Red
}

# ================== SAFE DISPLAY UTILITY ==================
def prepare_for_display(img_array):
    """Ensure image is safe for display in Streamlit: uint8 [0–255]."""
    arr = img_array.copy()

    # Case 1: float images
    if np.issubdtype(arr.dtype, np.floating):
        if arr.min() < 0 or arr.max() > 1.0:
            arr = (arr - arr.min()) / (np.ptp(arr) + 1e-8)   # np.ptp for NumPy ≥2.0
        arr = (arr * 255).astype(np.uint8)

    # Case 2: integer images
    elif np.issubdtype(arr.dtype, np.integer):
        if arr.max() > 255:
            arr = (255 * (arr - arr.min()) / (np.ptp(arr) + 1e-8)).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    return arr

# ================== LOAD DEEPLABV3+ MODEL ==================
from torchvision.models.segmentation import deeplabv3_resnet50   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_deeplab_model():
    num_classes = 9
    model = deeplabv3_resnet50(num_classes=num_classes, aux_loss=False, output_stride=16)
    weights_path = "D:\GeoSpatial_tools\landuse_classifier\deeplabv3_finetuned_RS_openearthmap_v2.pth"

    if not os.path.exists(weights_path):
        st.error(f"Model weights file not found at {os.path.abspath(weights_path)}")
        return None

    try:
        checkpoint = torch.load(weights_path, map_location=device)
        checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("aux_classifier")}
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

deeplab_model = load_deeplab_model()

# ================== MAIN APP ==================
def landuse_classifier(uploaded_files):
    st.header("🏞️ Land Use Classification")

    st.markdown("""
    Classify land use types in satellite imagery using:
    - NDVI-based classification
    - Spectral clustering
    - DeepLabV3+ (pretrained ML model)
    """)

    image_files = []
    for ext in ['tif', 'tiff', 'geotiff', 'jpg', 'jpeg', 'png']:
        image_files.extend(uploaded_files.get(ext, []))

    if not image_files:
        st.warning("Please upload satellite images first")
        return

    selected_file = st.selectbox("Select image for classification", [f.name for f in image_files])
    file = next(f for f in image_files if f.name == selected_file)

    method_options = ["Simple NDVI-based", "Spectral Clustering"]
    if deeplab_model is not None:
        method_options.append("DeepLabV3+ (ML Model)")

    method = st.radio("Choose classification approach:", method_options)

    st.sidebar.subheader("Classification Parameters")
    if method == "Simple NDVI-based":
        ndvi_threshold_water = st.sidebar.slider("Water threshold (NDVI < value)", -1.0, 0.0, -0.3)
        ndvi_threshold_vegetation = st.sidebar.slider("Vegetation threshold (NDVI > value)", 0.0, 1.0, 0.3)
    elif method == "Spectral Clustering":
        n_clusters = st.sidebar.slider("Number of clusters", 3, 10, 6)

    overlay = st.sidebar.checkbox("Show overlay with original image", value=True)

    if st.button("🔍 Classify Land Use", type="primary"):
        try:
            with st.spinner("Processing image..."):
                img_array, has_nir, red_band, nir_band, mask = load_image_for_classification(file)

                if method == "Simple NDVI-based":
                    classified = classify_by_ndvi(img_array, ndvi_threshold_water, ndvi_threshold_vegetation,
                                                  has_nir, red_band, nir_band, mask)
                elif method == "Spectral Clustering":
                    classified = classify_by_clustering(img_array, n_clusters, mask)
                elif method == "DeepLabV3+ (ML Model)":
                    classified = classify_with_deeplab(img_array, mask)

                display_classification_results(img_array, classified, method, overlay)

        except Exception as e:
            st.error(f"Classification failed: {str(e)}")

# ================== IMAGE LOADING ==================
def load_image_for_classification(file):
    has_nir = False
    red_band, nir_band = None, None
    mask = None
    try:
        if file.name.lower().endswith(('.tif', '.tiff', '.geotiff')):
            import rasterio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            try:
                with rasterio.open(tmp_path) as src:
                    img_array = src.read()  # (C, H, W)
                    img_array = np.transpose(img_array, (1, 2, 0))  # (H, W, C)

                    # --- Get valid-data mask ---
                    if src.nodata is not None:
                        mask = src.read_masks(1) > 0   # True = valid, False = nodata
                    else:
                        mask = np.ones((src.height, src.width), dtype=bool)

                    # Store NIR if available
                    if src.count >= 4:
                        has_nir = True
                        red_band = src.read(3).astype(float)
                        nir_band = src.read(4).astype(float)

                    # --- Fix channels for display ---
                    if img_array.shape[2] == 1:  # grayscale
                        img_array = np.repeat(img_array, 3, axis=2)
                    elif img_array.shape[2] == 2:  # 2-band (e.g., Red + NIR)
                        red_band = img_array[:, :, 0].astype(float)
                        nir_band = img_array[:, :, 1].astype(float)
                        has_nir = True
                        img_array = np.concatenate(
                            [img_array, np.zeros((img_array.shape[0], img_array.shape[1], 1))],
                            axis=2
                        )
                    elif img_array.shape[2] > 3:  # drop extras
                        img_array = img_array[:, :, :3]

            finally:
                os.unlink(tmp_path)
        else:
            img = Image.open(file)
            img_array = np.array(img)
            mask = np.ones(img_array.shape[:2], dtype=bool)  # no nodata in JPG/PNG
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] > 3:
                img_array = img_array[:, :, :3]

        return img_array, has_nir, red_band, nir_band, mask
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")

# ================== METHODS ==================
def classify_by_ndvi(img_array, water_threshold, veg_threshold, has_nir=False, red_band=None, nir_band=None, mask=None):
    if has_nir and red_band is not None and nir_band is not None:
        red = red_band
        nir = nir_band
    else:
        red = img_array[:, :, 0].astype(float)
        nir = img_array[:, :, 1].astype(float)

    ndvi = (nir - red) / (nir + red + 1e-10)
    classified = np.zeros(ndvi.shape, dtype=int)

    classified[ndvi < water_threshold] = 6
    classified[ndvi > veg_threshold] = 3
    mask_mid = (ndvi >= water_threshold) & (ndvi <= veg_threshold)
    classified[mask_mid & (ndvi > 0)] = 7
    classified[mask_mid & (ndvi <= 0)] = 1

    # Apply nodata mask
    if mask is not None:
        classified[~mask] = 0  # Background

    return classified


def classify_by_clustering(img_array, n_clusters, mask=None):
    from sklearn.cluster import KMeans
    h, w, c = img_array.shape
    pixels = img_array.reshape(-1, c)

    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    mask_flat = mask.flatten()

    valid_pixels = pixels[mask_flat]
    if len(valid_pixels) == 0:
        return np.zeros((h, w), dtype=int)

    sample_idx = np.random.choice(len(valid_pixels), min(50000, len(valid_pixels)), replace=False)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(valid_pixels[sample_idx])

    cluster_labels = np.zeros(len(pixels), dtype=int)
    cluster_labels[mask_flat] = kmeans.predict(valid_pixels)

    return cluster_labels.reshape(h, w)


def classify_with_deeplab(img_array, mask=None):
    if deeplab_model is None:
        return classify_by_clustering(img_array, 6, mask)

    h, w, _ = img_array.shape
    pil_img = Image.fromarray(prepare_for_display(img_array))

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = deeplab_model(input_tensor)["out"]
        prediction = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    prediction = np.array(
        Image.fromarray(prediction.astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
    )

    if mask is not None:
        prediction[~mask] = 0
    return prediction

# ================== DISPLAY ==================
def display_classification_results(original_img, classified, method, overlay=True):
    colored_classified = create_colored_classification_map(classified)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(prepare_for_display(original_img), caption="Input satellite image", use_container_width=True)

    with col2:
        st.subheader("Land Use Classification")
        if overlay:
            orig_255 = prepare_for_display(original_img)
            overlay_img = cv2.addWeighted(orig_255, 0.6, colored_classified, 0.4, 0)
            st.image(overlay_img, caption=f"{method} (Overlay)", use_container_width=True)
        else:
            st.image(colored_classified, caption=f"{method}", use_container_width=True)

    unique_classes, counts = np.unique(classified, return_counts=True)
    total_pixels = classified.size
    stats_data = []
    for class_id, count in zip(unique_classes, counts):
        class_name = LAND_USE_CLASSES.get(class_id, f"Class {class_id}")
        percentage = (count / total_pixels) * 100
        stats_data.append({
            'Land Use Type': class_name,
            'Pixels': count,
            'Percentage': percentage
        })
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df.style.format({"Percentage": "{:.2f}%"}), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    class_names = [LAND_USE_CLASSES.get(cls, f"Class {cls}") for cls in unique_classes]
    percentages = [(count / total_pixels) * 100 for count in counts]
    bars = ax.bar(class_names, percentages)
    for bar, class_id in zip(bars, unique_classes):
        bar.set_color(np.array(CLASS_COLORS.get(class_id, [128, 128, 128])) / 255.0)
    ax.set_ylabel("Percentage of Area")
    ax.set_title("Land Use Distribution")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("💾 Export Results"):
        from PIL import Image as PILImage
        classified_img = PILImage.fromarray(colored_classified.astype(np.uint8))
        import io
        buf = io.BytesIO()
        classified_img.save(buf, format='PNG')
        st.download_button("Download Classification Map", buf.getvalue(),
                           file_name="land_use_classification.png", mime="image/png")
        csv = stats_df.to_csv(index=False)
        st.download_button("Download Statistics", data=csv,
                           file_name="land_use_statistics.csv", mime="text/csv")

def create_colored_classification_map(classified):
    h, w = classified.shape
    colored_map = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored_map[classified == class_id] = color
    return colored_map
