import os
from functools import lru_cache

from PIL import Image, ImageDraw

from landuse_classifier.classifier import load_image_for_classification
from utils.visualization import prepare_for_display

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install at runtime

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 165, 0), (128, 0, 128), (0, 128, 128), (128, 128, 0)]


@lru_cache(maxsize=1)
def load_model():
    from ultralytics import YOLO
    # ponytail: stock COCO yolov8n; swap in a satellite-trained (e.g. DOTA) model for real aerial classes
    return YOLO(os.path.join(os.path.dirname(__file__), 'yolov8n.pt'))


def detect(file, conf=0.25, iou=0.45):
    """Run YOLO on an uploaded image/raster. Returns (annotated PIL image, detections list)."""
    img = Image.fromarray(prepare_for_display(load_image_for_classification(file)[0])).convert("RGB")
    result = load_model().predict(img, conf=conf, iou=iou, imgsz=640, verbose=False)[0]

    draw = ImageDraw.Draw(img)
    width = max(2, img.width // 400)
    detections = []
    for box in result.boxes:
        class_id = int(box.cls)
        name, confidence = result.names[class_id], float(box.conf)
        bbox = [round(v, 1) for v in box.xyxy[0].tolist()]
        detections.append({"class_name": name, "confidence": round(confidence, 3), "bbox": bbox})
        color = COLORS[class_id % len(COLORS)]
        draw.rectangle(bbox, outline=color, width=width)
        draw.text((bbox[0] + 3, max(0, bbox[1] - 14)), f"{name} {confidence:.2f}", fill=color)
    return img, detections
