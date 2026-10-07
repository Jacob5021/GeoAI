import numpy as np

from ndvi_viewer.ndvi_processor import calculate_ndvi
from utils.visualization import prepare_for_display
from utils.geospatial_utils import read_band_wgs84, to_geotiff_bytes

import io
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin


def _utm_tif():
    """10x10 uint16 raster in UTM 43N (near Bengaluru) with a nodata corner."""
    data = np.arange(100, dtype=np.uint16).reshape(10, 10) + 1
    data[0, 0] = 0
    mem = MemoryFile()
    with mem.open(driver="GTiff", height=10, width=10, count=1, dtype="uint16", nodata=0,
                  crs="EPSG:32643", transform=from_origin(780000, 1440000, 100, 100)) as dst:
        dst.write(data, 1)
    return io.BytesIO(mem.read())


def test_calculate_ndvi():
    red = np.array([[0.1, 0.0, np.nan]])
    nir = np.array([[0.5, 0.0, 0.4]])
    ndvi = calculate_ndvi(red, nir)
    assert np.isclose(ndvi[0, 0], 0.4 / 0.6)
    assert ndvi[0, 1] == 0          # 0/0 guarded
    assert np.isnan(ndvi[0, 2])     # nodata stays NaN


def test_prepare_for_display():
    u16 = np.array([[0, 5000, 10000]], dtype=np.uint16)   # 16-bit reflectance
    out = prepare_for_display(u16)
    assert out[0, 0] == 0 and out[0, 2] == 255 and 126 <= out[0, 1] <= 128
    assert prepare_for_display(np.array([[0.0, 1.0]])).tolist() == [[0, 255]]
    assert prepare_for_display(np.array([[np.nan, 2.0, 4.0]])).dtype == np.uint8
    rgb = np.array([[10, 200]], dtype=np.uint8)
    assert prepare_for_display(rgb).tolist() == [[10, 200]]  # 8-bit untouched


def test_read_band_wgs84():
    with rasterio.open(_utm_tif()) as src:
        data, b = read_band_wgs84(src)
    assert data.dtype == np.float32 and np.isnan(data).any()      # nodata -> NaN
    assert 77 < b.left < b.right < 78 and 12 < b.bottom < b.top < 14  # degrees, not metres
    assert 1 <= np.nanmin(data) and np.nanmax(data) <= 100


def test_to_geotiff_bytes():
    f = _utm_tif()
    ndvi = np.full((10, 10), 0.5, dtype=np.float32)
    with rasterio.open(io.BytesIO(to_geotiff_bytes(ndvi, f, nodata=np.nan))) as out:
        assert out.crs == "EPSG:32643" and out.transform == from_origin(780000, 1440000, 100, 100)
        assert np.allclose(out.read(1), 0.5)
    assert f.tell() == 0  # source rewound for the next reader


def test_read_band_wgs84_downsamples():
    with rasterio.open(_utm_tif()) as src:
        data, _ = read_band_wgs84(src, max_size=4)
    assert max(data.shape) <= 6  # 10px -> ~4px, plus reprojection padding


def _rgba_tif():
    """Grayscale-as-RGBA export: identical R=G=B, alpha 0 on the left half."""
    gray = np.tile(np.arange(10, dtype=np.uint8) * 20, (10, 1))
    alpha = np.full((10, 10), 255, dtype=np.uint8)
    alpha[:, :5] = 0
    mem = MemoryFile()
    with mem.open(driver="GTiff", height=10, width=10, count=4, dtype="uint8", photometric="RGB",
                  alpha="YES", crs="EPSG:32644", transform=from_origin(0, 0, 1, 1)) as dst:
        dst.write(np.stack([gray, gray, gray, alpha]))
    f = io.BytesIO(mem.read())
    f.name = "rgba.tif"
    return f


def test_landuse_ignores_alpha_band():
    from landuse_classifier.classifier import load_image_for_classification, classify_by_clustering
    img, has_nir, _, _, mask = load_image_for_classification(_rgba_tif())
    assert img.shape == (10, 10, 3) and not has_nir     # alpha is not NIR
    assert not mask[:, :5].any() and mask[:, 5:].all()  # alpha 0 -> no data
    labels = classify_by_clustering(img, 2, mask)
    assert (labels[:, :5] == 0).all() and (labels[:, 5:] >= 1).all()


if __name__ == "__main__":
    test_calculate_ndvi()
    test_prepare_for_display()
    test_read_band_wgs84()
    test_to_geotiff_bytes()
    test_read_band_wgs84_downsamples()
    test_landuse_ignores_alpha_band()
    print("ok")
