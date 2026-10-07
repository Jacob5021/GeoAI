import base64
import io

import numpy as np
from matplotlib import colormaps
from PIL import Image


def prepare_for_display(img_array):
    """Scale any numeric image to uint8 [0-255] for display/PIL/OpenCV (NaN -> 0)."""
    src = np.asarray(img_array)
    arr = np.nan_to_num(src.astype(float))
    if np.issubdtype(src.dtype, np.floating) and arr.min() >= 0 and arr.max() <= 1:
        arr = arr * 255
    elif np.issubdtype(src.dtype, np.floating) or arr.min() < 0 or arr.max() > 255:
        arr = (arr - arr.min()) / (np.ptp(arr) + 1e-8) * 255
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def colorize(values, cmap, vmin, vmax):
    """Map a 2D float array to RGBA uint8 with a matplotlib colormap; NaN becomes transparent."""
    norm = np.clip((values - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgba = (colormaps[cmap](np.nan_to_num(norm)) * 255).astype(np.uint8)
    rgba[np.isnan(values), 3] = 0
    return rgba


def png_bytes(image):
    """Encode a PIL image or uint8 array as PNG bytes."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def png_data_url(image, max_size=1600):
    """PNG data URL for display, downscaled so the longer side is at most max_size."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if max(image.size) > max_size:
        image = image.copy()
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    return "data:image/png;base64," + base64.b64encode(png_bytes(image)).decode()
