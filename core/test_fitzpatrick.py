import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
sys.path.append(r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models")
from model import get_resnet50_model


# 1. Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_WEIGHTS = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\melanoma_finetuned.pth"
DATA_DIR = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\fitz_ham10000_subset"

# Only testing Tone 1 and Tone 6
TONES_TO_TEST = ['1', '6']
IMAGES_PER_TONE = 20

# 2. HAM10000 Class Mapping 
# Our ResNet outputs 0-6. The filenames in the new fitz folder are prefixed with the correct string label
idx_to_class = {
    0: 'akiec', 
    1: 'bcc', 
    2: 'bkl', 
    3: 'df', 
    4: 'mel', 
    5: 'nv', 
    6: 'vasc'
}

# Image Preprocessing (matching the training pipeline)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_tone_bias():
    print(f"--- Running Fitzpatrick Bias Evaluation on {DEVICE} ---")
    
    # Load Fine-Tuned Model
    model, _ = get_resnet50_model(num_classes=7)
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    else:
        print(f"ERROR: Cannot find finetuned weights at {MODEL_WEIGHTS}")
        return
        
    model = model.to(DEVICE)
    model.eval()
    
    # Dictionary to track results
    results = {'1': {'correct': 0, 'total': 0}, '6': {'correct': 0, 'total': 0}}
    
    for tone in TONES_TO_TEST:
        tone_dir = os.path.join(DATA_DIR, tone)
        if not os.path.exists(tone_dir):
            print(f"Missing directory: {tone_dir}")
            continue
            
        print(f"\nEvaluating Skin Tone Type {tone}...")
        
        # Get all images in this tone folder
        images = [f for f in os.listdir(tone_dir) if f.endswith('.jpg')]
        
        # Limit to the requested randomly sampled subset (or max available)
        images = images[:IMAGES_PER_TONE]
        
        for img_name in images:
            img_path = os.path.join(tone_dir, img_name)
            
            # Extract actual label from the prefix (e.g. "mel_0a7d0e3f4df5...jpg")
            actual_label = img_name.split('_')[0]
            
            try:
                # Bypass Healthy checks as requested earlier for pure evaluation
                # Using pure PIL image load since we don't need cv2 preprocessing right now
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, preds_tensor = torch.max(outputs, 1)
                    pred_idx = preds_tensor.item()
                    
                predicted_label = idx_to_class[pred_idx]
                
                # Check accuracy
                match = predicted_label == actual_label
                if match:
                    results[tone]['correct'] += 1
                
                results[tone]['total'] += 1
                
                status_str = "[WIN]" if match else "[FAIL]"
                print(f"  {status_str} | Pred: {predicted_label.upper().ljust(5)} | Actual: {actual_label.upper().ljust(5)} | Hash: {img_name.split('_')[1][:8]}...")
                
            except Exception as e:
                print(f"  Error processing {img_name}: {e}")

    # Final Report
    print("\n" + "="*40)
    print("FITZPATRICK SKIN TONE BIAS REPORT")
    print("="*40)
    for tone in TONES_TO_TEST:
        total = results[tone]['total']
        if total > 0:
            correct = results[tone]['correct']
            acc = (correct / total) * 100
            print(f"Skin Tone {tone}: {acc:.1f}% Accuracy ({correct}/{total})")
        else:
            print(f"Skin Tone {tone}: No images evaluated.")
    print("="*40)

if __name__ == "__main__":
    evaluate_tone_bias()
