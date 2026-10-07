import os
from functools import lru_cache

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from PIL import Image

from utils.visualization import prepare_for_display

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

# Distinct colours for unlabelled k-means clusters (1..10)
CLUSTER_COLORS = [[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
                  [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207]]

METHODS = ["Simple NDVI-based", "Spectral Clustering", "DeepLabV3+ (ML Model)"]
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "deeplabv3_finetuned_RS_openearthmap_v2.pth")


# ================== DEEPLABV3+ MODEL ==================
@lru_cache(maxsize=1)
def load_deeplab_model():
    """DeepLabV3 fine-tuned on OpenEarthMap, or None when the weights file is absent."""
    if not os.path.exists(WEIGHTS_PATH):
        return None
    import torch
    from torchvision.models.segmentation import deeplabv3_resnet50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet50(num_classes=9, aux_loss=False, output_stride=16)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("aux_classifier")}
    model.load_state_dict(checkpoint, strict=False)
    return model.to(device).eval()


def available_methods():
    return METHODS if os.path.exists(WEIGHTS_PATH) else METHODS[:2]


# ================== IMAGE LOADING ==================
def load_image_for_classification(file):
    """Return (img HxWx3, has_nir, red_band, nir_band, valid_mask) from an uploaded image/raster."""
    has_nir = False
    red_band, nir_band = None, None
    try:
        if file.name.lower().endswith(('.tif', '.tiff', '.geotiff')):
            with MemoryFile(file.getvalue()) as mem, mem.open() as src:
                # Alpha bands are masks, not spectral data (RGBA exports are common)
                data_bands = [i + 1 for i, c in enumerate(src.colorinterp) if c != ColorInterp.alpha]
                img_array = np.transpose(src.read(data_bands), (1, 2, 0))  # (H, W, C)

                # Valid-data mask from nodata, alpha band or internal mask
                mask = src.dataset_mask() > 0

                if len(data_bands) >= 4:
                    has_nir = True
                    red_band = src.read(data_bands[2]).astype(float)
                    nir_band = src.read(data_bands[3]).astype(float)

            if img_array.shape[2] == 1:  # grayscale
                img_array = np.repeat(img_array, 3, axis=2)
            elif img_array.shape[2] == 2:  # 2-band (e.g., Red + NIR)
                red_band = img_array[:, :, 0].astype(float)
                nir_band = img_array[:, :, 1].astype(float)
                has_nir = True
                img_array = np.concatenate([img_array, np.zeros_like(img_array[:, :, :1])], axis=2)
            elif img_array.shape[2] > 3:  # drop extras
                img_array = img_array[:, :, :3]
        else:
            img_array = np.array(Image.open(file))
            mask = np.ones(img_array.shape[:2], dtype=bool)  # no nodata in JPG/PNG
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] > 3:
                img_array = img_array[:, :, :3]

        return img_array, has_nir, red_band, nir_band, mask
    except Exception as e:
        raise ValueError(f"Error loading image: {e}") from e


# ================== METHODS ==================
def classify_by_ndvi(img_array, water_threshold, veg_threshold, has_nir=False, red_band=None, nir_band=None, mask=None):
    if has_nir and red_band is not None and nir_band is not None:
        red, nir = red_band, nir_band
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

    sample_idx = np.random.default_rng(42).choice(len(valid_pixels), min(50000, len(valid_pixels)), replace=False)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(valid_pixels[sample_idx])

    cluster_labels = np.zeros(len(pixels), dtype=int)  # 0 = no data
    cluster_labels[mask_flat] = kmeans.predict(valid_pixels) + 1
    return cluster_labels.reshape(h, w)


def classify_with_deeplab(img_array, mask=None):
    import torch
    import torchvision.transforms as T
    model = load_deeplab_model()
    if model is None:
        raise ValueError("DeepLabV3+ weights not found")

    h, w, _ = img_array.shape
    pil_img = Image.fromarray(prepare_for_display(img_array))
    device = next(model.parameters()).device
    input_tensor = T.Compose([T.Resize((512, 512)), T.ToTensor()])(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = torch.argmax(model(input_tensor)["out"].squeeze(), dim=0).cpu().numpy()

    prediction = np.array(Image.fromarray(prediction.astype(np.uint8)).resize((w, h), resample=Image.NEAREST))
    if mask is not None:
        prediction[~mask] = 0
    return prediction


def classify(file, method, water_threshold=-0.3, veg_threshold=0.3, n_clusters=6):
    """Run one method. Returns (img, classified, names, colors, has_nir)."""
    img, has_nir, red, nir, mask = load_image_for_classification(file)
    if method == "Simple NDVI-based":
        classified = classify_by_ndvi(img, water_threshold, veg_threshold, has_nir, red, nir, mask)
        return img, classified, LAND_USE_CLASSES, CLASS_COLORS, has_nir
    if method == "Spectral Clustering":
        classified = classify_by_clustering(img, n_clusters, mask)
        # k-means clusters are unlabelled: don't dress them up as land-use classes
        names = {0: "No data", **{k: f"Cluster {k}" for k in range(1, n_clusters + 1)}}
        colors = {0: [0, 0, 0], **{k: CLUSTER_COLORS[(k - 1) % len(CLUSTER_COLORS)] for k in range(1, n_clusters + 1)}}
        return img, classified, names, colors, has_nir
    if method == "DeepLabV3+ (ML Model)":
        return img, classify_with_deeplab(img, mask), LAND_USE_CLASSES, CLASS_COLORS, has_nir
    raise ValueError(f"Unknown method: {method}")


def create_colored_classification_map(classified, colors=CLASS_COLORS):
    colored_map = np.zeros((*classified.shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        colored_map[classified == class_id] = color
    return colored_map
