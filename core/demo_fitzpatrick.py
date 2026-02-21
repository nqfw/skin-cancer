import os
import cv2
import torch
import numpy as np
import random
import sys
import torchvision.transforms as transforms
from PIL import Image

# Modify paths to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import get_resnet50_model
from core.dullrazor import apply_dullrazor
from core.gradcam_engine import generate_cam

# Import MST estimator
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "skintone"))
from skintone import estimate_skin_tone

def load_random_images(data_dir, total_samples=40):
    """
    Randomly samples images from all skin tone bins.
    """
    all_images = []
    
    # 1 to 6
    for tone in range(1, 7):
        tone_dir = os.path.join(data_dir, str(tone))
        if os.path.exists(tone_dir):
            for f in os.listdir(tone_dir):
                if f.endswith('.jpg'):
                    all_images.append({
                        'path': os.path.join(tone_dir, f),
                        'true_label': f.split('_')[0],
                        'ground_truth_tone': tone
                    })
                    
    if len(all_images) == 0:
        return []
        
    sample_size = min(total_samples, len(all_images))
    return random.sample(all_images, sample_size)

def demo_fitzpatrick_gradcam():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # We will test using the HAM10000 finetuned model, since train_fitzpatrick may not be completely done
    weights_path = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\melanoma_finetuned.pth"
    data_dir = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\fitz_ham10000_subset"

    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return
        
    # 2. Setup Model
    print("Loading baseline model for visualization...")
    model, _ = get_resnet50_model(num_classes=7)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    target_layer = model.layer4[-1]
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    idx_to_class = {0:'akiec', 1:'bcc', 2:'bkl', 3:'df', 4:'mel', 5:'nv', 6:'vasc'}

    # 3. Load Images
    images_to_test = load_random_images(data_dir, total_samples=40)
    if not images_to_test:
        print("No images found in the fitzpatrick subset directory.")
        return

    print(f"Loaded {len(images_to_test)} images for loose testing.")
    print("\nControls:")
    print("- Press 'Enter' or 'Space' to go to the next image")
    print("- Press 'Q' or 'Escape' to quit the demo\n")

    for img_data in images_to_test:
        # Load and verify
        img_path = img_data['path']
        true_label = img_data['true_label']
        fitz_tone = img_data['ground_truth_tone']
        
        orig_img = cv2.imread(img_path)
        if orig_img is None: continue
        
        orig_img_resized = cv2.resize(orig_img, (400, 300))
        
        # 4. Predict MST Tone
        predicted_mst = estimate_skin_tone(orig_img_resized)
        
        # 5. Preprocess (DullRazor)
        clean_img_bgr = apply_dullrazor(orig_img_resized)
        clean_img_rgb = cv2.cvtColor(clean_img_bgr, cv2.COLOR_BGR2RGB)
        
        # 6. PyTorch Inference
        input_tensor = transform(clean_img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
        pred_label = idx_to_class[preds[0].item()]
        
        # 7. Generate GradCAM
        clean_img_224 = cv2.resize(clean_img_bgr, (224, 224))
        clean_img_float = np.float32(clean_img_224) / 255.0
        
        heatmap_bgr = generate_cam(model, target_layer, input_tensor, clean_img_float)
        heatmap_bgr = cv2.resize(heatmap_bgr, (400, 300))
        
        # 8. Render
        # Layout: Original | Cleaned | Heatmap
        combined = np.hstack((orig_img_resized, clean_img_bgr, heatmap_bgr))
        
        # UI Text Overlays
        # Top banner: Diagnosis
        disease_color = (0, 255, 0) if pred_label == true_label else (0, 0, 255)
        cv2.putText(combined, f"Pred: {pred_label.upper()} | True: {true_label.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, disease_color, 2)
                   
        # Bottom banner: Skin Tone Analysis
        # Compare actual Fitzpatrick group (1-6 scale) to predicted MST (1-10 scale)
        tone_text = f"Group Tone: Type {fitz_tone} | Predicted MST: {predicted_mst}/10"
        cv2.putText(combined, tone_text, 
                   (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
        cv2.imshow("Fitzpatrick Evaluation Demo", combined)
        print(f"Displaying: {os.path.basename(img_path)} | Fitz: {fitz_tone} -> MST: {predicted_mst} | L: {true_label.upper()} -> P: {pred_label.upper()}")
        
        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord('q'): # ESC or Q
            break
            
    cv2.destroyAllWindows()
    print("Demo completed.")

if __name__ == "__main__":
    demo_fitzpatrick_gradcam()
