import os
import streamlit as st
from utils.visualization import page_header, empty_state
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import cv2
os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install at runtime
from ultralytics import YOLO
import tempfile
import pandas as pd
import tifffile
from utils.visualization import prepare_for_display

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 165, 0), (128, 0, 128), (0, 128, 128), (128, 128, 0)]

@st.cache_resource
def load_model():
    # ponytail: stock COCO yolov8n; swap in a satellite-trained (e.g. DOTA) model for real aerial classes
    return YOLO(os.path.join(os.path.dirname(__file__), 'yolov8n.pt'))

def satellite_detector(uploaded_files):
    page_header("Object Detection", "YOLOv8 object detection on imagery (COCO classes).")
    
    # Check for suitable files
    image_files = []
    for ext in ['tif', 'tiff', 'geotiff', 'jpg', "jpeg", 'png']:
        image_files.extend(uploaded_files.get(ext, []))
    
    if not image_files:
        empty_state("No imagery yet.")
        return
    
    # File selection
    selected_file = st.selectbox("Select image", [f.name for f in image_files])
    file = next(f for f in image_files if f.name == selected_file)
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold", 
        0.1, 1.0, 0.25,
        help="Minimum detection confidence"
    )
    iou_thresh = st.sidebar.slider(
        "IOU Threshold", 
        0.1, 0.9, 0.45,
        help="Intersection over Union threshold"
    )
    
    if st.button("Detect Objects", type="primary"):
        try:
            with st.spinner("Loading model..."):
                model = load_model()
            
            with st.spinner("Processing image..."):
                # Save uploaded file to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                    tmp.write(file.getvalue())
                    img_path = tmp.name
                
                # ---------- Ensure RGB for YOLO ----------
                try:
                    img = Image.open(img_path)
                except Exception:
                    # Fallback for GeoTIFF / multispectral
                    arr = tifffile.imread(img_path)

                    if arr.ndim == 2:  
                        # Grayscale → replicate 3 times
                        arr = np.stack([arr]*3, axis=-1)
                    elif arr.ndim == 3:
                        if arr.shape[0] < arr.shape[-1]:  # (C, H, W) → (H, W, C)
                            arr = np.moveaxis(arr, 0, -1)
                        # Multispectral → take first 3 bands
                        arr = arr[..., :3] if arr.shape[-1] >= 3 else np.repeat(arr[..., :1], 3, axis=-1)

                    img = Image.fromarray(prepare_for_display(arr))

                # Ensure RGB mode
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Save back for YOLO
                img.save(img_path)
                # -----------------------------------------

                # Run inference
                results = model.predict(
                    img_path,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    imgsz=640
                )
                
                # Process results
                draw = ImageDraw.Draw(img)
                detections = []
                
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = result.names[class_id]
                        confidence = float(box.conf)
                        bbox = box.xyxy[0].tolist()
                        
                        detections.append({
                            'class_name': class_name,
                            'bbox': bbox,
                            'confidence': confidence
                        })
                        
                        # Draw bounding box
                        color = COLORS[class_id % len(COLORS)]
                        draw.rectangle(bbox, outline=color, width=3)
                        
                        # Add label
                        label = f"{class_name}: {confidence:.2f}"
                        draw.text((bbox[0], bbox[1] - 15), label, fill=color)
                
                os.unlink(img_path)  # Clean up temp file
                
                # Display results
                st.image(img, caption=f"Detected {len(detections)} objects", width="stretch")
                
                # Detection summary
                st.subheader("Detection Summary")
                counts = pd.Series([d['class_name'] for d in detections], dtype=object).value_counts()
                cols = st.columns(4)
                for i, (name, count) in enumerate(counts.items()):
                    cols[i % 4].metric(name, int(count))
                
                # Download options
                with st.expander("💾 Export Results"):
                    # Image
                    output_img = BytesIO()
                    img.save(output_img, format='PNG')
                    st.download_button(
                        "Download Annotated Image",
                        output_img.getvalue(),
                        file_name=f"detected_{os.path.splitext(file.name)[0]}.png",
                        mime="image/png"
                    )
                    
                    # CSV
                    if detections:
                        df = pd.DataFrame(detections)
                        csv = df[['class_name', 'confidence', 'bbox']].to_csv(index=False)
                        st.download_button(
                            "Download Detection Data",
                            data=csv,
                            file_name="detections.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Detection failed: {str(e)}")
            if 'img_path' in locals() and os.path.exists(img_path):
                os.unlink(img_path)
