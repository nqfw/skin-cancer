import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import sys
import os

from gradcam_engine import generate_cam
from dullrazor import apply_dullrazor
from skin import process_image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import get_resnet50_model

def run_gradcam_demo(image_path):
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
        
    img = cv2.resize(img, (400, 300))
    
    # 2. Clean image (DullRazor + Ruler Removal)
    print("Pre-processing Image (Hair/Ruler removal)...")
    clean_img = apply_dullrazor(img)
    
    # 3. Check for skin (Optional safety check)
    is_skin, pct, _ = process_image(clean_img)
    if not is_skin:
        print(f"Warning: Low skin coverage ({pct:.1f}%). Generating heatmap anyway...")

    # 4. Prepare for PyTorch Model
    clean_img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
    
    # Standard ImageNet normalization for IMAGENET1K_V2 weights
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(clean_img_rgb).unsqueeze(0) # 1x3x224x224
    
    # 5. Load Model
    print("Loading Pretrained ResNet50 (IMAGENET1K_V2)...")
    model, _ = get_resnet50_model()
    model.eval()
    
    # 6. Generate Grad-CAM Match
    # For ResNet50, the last conv layer is layer4[-1]
    target_layer = model.layer4[-1]
    
    print("Inferring Heatmap...")
    # Generate the heatmap. We pass the 224x224 BGR original so the cam size matches exactly!
    clean_img_224 = cv2.resize(clean_img, (224, 224))
    clean_img_float = np.float32(clean_img_224) / 255.0
    heatmap_bgr = generate_cam(model, target_layer, input_tensor, clean_img_float)
    
    # Resize the heatmap back up to 400x300 so it matches the other images in our display loop
    heatmap_bgr = cv2.resize(heatmap_bgr, (400, 300))
    
    # 7. Create Side-by-Side Visualization
    # [Original | Cleaned | Grad-CAM Heatmap]
    combined = np.hstack((img, clean_img, heatmap_bgr))
    
    # Add text overlay with results
    text = f"ResNet50 Heatmap | Skin Cover: {pct:.1f}%"
    cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 8. Show Image
    print("Done! Displaying results.")
    cv2.imshow("Trust Layout (Original | Cleaned | Grad-CAM)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python core/demo_gradcam.py <path_to_image>")
        sys.exit(1)
        
    run_gradcam_demo(sys.argv[1])
