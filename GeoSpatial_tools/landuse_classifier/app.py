import streamlit as st
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms

# Updated class labels and colors for OpenEarthMap v2
CLASS_LABELS = {
    0: "Background",
    1: "Bareland",
    2: "Rangeland",
    3: "Developed Space",
    4: "Road",
    5: "Tree",
    6: "Water",
    7: "Agriculture Land",
    8: "Building"
}

CLASS_COLORS = {
    0: [0, 0, 0],        # Background - Black
    1: [128, 0, 0],      # Bareland - Dark Red
    2: [0, 255, 36],     # Rangeland - Bright Green
    3: [148, 148, 148],  # Developed Space - Gray
    4: [255, 255, 255],  # Road - White
    5: [34, 97, 38],     # Tree - Dark Green
    6: [0, 69, 255],     # Water - Blue
    7: [75, 181, 73],    # Agriculture Land - Light Green
    8: [222, 31, 7]      # Building - Orange-Red
}

# Load PyTorch model
@st.cache_resource
def load_model():
    """Load the custom PyTorch DeepLabV3+ model"""
    # Import the model architecture (you'll need to have this in your project)
    from model.deeplabv3plus import DeepLabV3Plus  # Adjust import path as needed
    
    model = DeepLabV3Plus(
        backbone="resnet50",
        num_classes=9,  # 9 classes in OpenEarthMap v2
        output_stride=16,
        pretrained_backbone=False
    )
    
    # Load the pretrained weights
    checkpoint = torch.load("deeplabv3_finetuned_RS_openearthmap_v2.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

def landuse_classifier(uploaded_files):
    st.header("🌍 Satellite Land Use Classifier")
    st.markdown("""
    Classify satellite imagery using a fine-tuned DeepLabV3+ model 
    (trained on OpenEarthMap dataset)
    """)
    
    # Check for suitable files
    image_files = []
    for ext in ['tif', 'tiff', 'geotiff', 'jpg', 'png']:
        image_files.extend(uploaded_files.get(ext, []))
    
    if not image_files:
        st.warning("Please upload satellite images first")
        return
    
    # File selection
    selected_file = st.selectbox("Select image", [f.name for f in image_files])
    file = next(f for f in image_files if f.name == selected_file)
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    confidence_thresh = st.sidebar.slider(
        "Confidence Threshold", 
        0.1, 1.0, 0.5,
        help="Minimum confidence for classification"
    )
    
    if st.button("Classify Land Use", type="primary"):
        try:
            with st.spinner("Loading pre-trained model..."):
                model = load_model()
            
            with st.spinner("Processing image..."):
                # Load and preprocess image
                img = Image.open(file).convert("RGB")
                img_array = np.array(img)
                
                # Convert to PyTorch tensor and normalize
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                input_tensor = transform(img).unsqueeze(0)  # Add batch dimension
                
                # Run inference
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_mask = torch.argmax(output, dim=1).squeeze(0).numpy()
                
                # Create visualization
                classified_img = create_classified_image(pred_mask)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original Image", use_column_width=True)
                with col2:
                    st.image(classified_img, caption="Classified Image", use_column_width=True)
                
                # Show statistics
                show_class_distribution(pred_mask)
                
                # Download options
                with st.expander("💾 Download Results"):
                    # Classified image
                    output_img = BytesIO()
                    Image.fromarray(classified_img).save(output_img, format='PNG')
                    st.download_button(
                        "Download Classification",
                        output_img.getvalue(),
                        file_name=f"classified_{file.name.split('.')[0]}.png",
                        mime="image/png"
                    )
                    
                    # Class probabilities (convert output to numpy first)
                    output_probs = torch.softmax(output, dim=1).squeeze(0).numpy()
                    output_csv = BytesIO()
                    np.savetxt(output_csv, output_probs.reshape(-1, output_probs.shape[-1]), delimiter=",")
                    st.download_button(
                        "Download Class Probabilities", 
                        output_csv.getvalue(),
                        file_name=f"probabilities_{file.name.split('.')[0]}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Classification failed: {str(e)}")

def create_classified_image(class_map):
    """Convert class indices to colored image"""
    height, width = class_map.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        colored[class_map == class_id] = color
        
    return colored

def show_class_distribution(class_map):
    """Display class statistics"""
    st.subheader("Classification Statistics")
    
    counts = {CLASS_LABELS[class_id]: np.sum(class_map == class_id) 
             for class_id in CLASS_LABELS}
    total = class_map.size
    
    # Display metrics
    cols = st.columns(len(CLASS_LABELS))
    for idx, (class_id, label) in enumerate(CLASS_LABELS.items()):
        with cols[idx]:
            percent = (counts[label] / total) * 100
            st.metric(
                label,
                f"{percent:.1f}%",
                help=f"{counts[label]:,} pixels"
            )
    
    # Show pie chart
    fig, ax = plt.subplots()
    ax.pie(
        counts.values(),
        labels=counts.keys(),
        colors=[np.array(CLASS_COLORS[cid])/255 for cid in CLASS_LABELS.keys()],
        autopct='%1.1f%%'
    )
    st.pyplot(fig)