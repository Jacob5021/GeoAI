import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import cv2
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import tifffile

# Real classes from a satellite-trained YOLO model
OBJECT_CLASSES = {
    0: "Airplane",
    1: "Ship",
    2: "Storage Tank",
    3: "Baseball Diamond",
    4: "Tennis Court",
    5: "Vehicle",
    6: "Building",
    7: "Road",
    8: "Bridge",
    9: "Harbor"
}

OBJECT_COLORS = {
    0: (255, 0, 0),    # Red
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Blue
    3: (255, 255, 0),  # Yellow
    4: (255, 0, 255),  # Magenta
    5: (0, 255, 255),  # Cyan
    6: (255, 165, 0),  # Orange
    7: (128, 0, 128),  # Purple
    8: (0, 128, 128),  # Teal
    9: (128, 128, 0)   # Olive
}

@st.cache_resource
def load_model():
    return YOLO('D:/GeoSpatial_tools/satellite_detection/yolov8n.pt')

def satellite_detector(uploaded_files):
    st.header("🛰️ Satellite Object Detection")
    st.markdown("""
    Detect objects in satellite imagery using YOLO trained on satellite datasets
    """)
    
    # Check for suitable files
    image_files = []
    for ext in ['tif', 'tiff', 'geotiff', 'jpg', "jpeg", 'png']:
        image_files.extend(uploaded_files.get(ext, []))
    
    if not image_files:
        st.warning("Please upload satellite images first")
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
                    elif arr.ndim == 3 and arr.shape[-1] > 3:
                        # Multispectral → take first 3 bands
                        arr = arr[..., :3]

                    img = Image.fromarray(arr.astype(np.uint8))

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
                        confidence = float(box.conf)
                        bbox = box.xyxy[0].tolist()
                        
                        detections.append({
                            'class': class_id,
                            'bbox': bbox,
                            'confidence': confidence
                        })
                        
                        # Draw bounding box
                        color = OBJECT_COLORS.get(class_id, (255,255,255))
                        draw.rectangle(bbox, outline=color, width=3)
                        
                        # Add label
                        label = f"{OBJECT_CLASSES.get(class_id, 'Unknown')}: {confidence:.2f}"
                        draw.text((bbox[0], bbox[1] - 15), label, fill=color)
                
                os.unlink(img_path)  # Clean up temp file
                
                # Display results
                st.image(img, caption=f"Detected {len(detections)} objects", use_column_width=True)
                
                # Detection summary
                st.subheader("Detection Summary")
                counts = {name: 0 for name in OBJECT_CLASSES.values()}
                for det in detections:
                    counts[OBJECT_CLASSES.get(det['class'], "Unknown")] += 1
                
                cols = st.columns(4)
                for i, (name, count) in enumerate(counts.items()):
                    if count > 0:
                        cols[i % 4].metric(name, count)
                
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
                        df['class_name'] = df['class'].map(OBJECT_CLASSES)
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
