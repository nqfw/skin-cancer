import fiftyone as fo
import pandas as pd
import os
import glob
import torch
from torchvision import transforms
from PIL import Image
import sys

def run_fiftyone_audit():
    print("Initializing FiftyOne Audit Pipeline with Model Inference...")
    
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "HAM10000 dataset", "HAM10000_images_part_1")
    csv_path = os.path.join(project_root, "data", "HAM10000 dataset", "HAM10000_metadata.csv")
    weights_path = os.path.join(project_root, "models", "melanoma_finetuned.pth")
    
    # 1. Setup Model
    sys.path.append(os.path.join(project_root, "models"))
    try:
        from model import get_resnet50_model
    except ImportError:
        print("Error: Could not import get_resnet50_model.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading ResNet50 model on {device}...")
    model, _ = get_resnet50_model(num_classes=7)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"Error: Weights not found at {weights_path}")
        return
    model = model.to(device)
    model.eval()
    
    # Preprocessing to match training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # EXACT labeling map used to train the HAM10000 model in train.py!
    idx_to_class = {0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df'}

    # 2. Load Metadata
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 3. Get first 200 images for speedy audit
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))[:200]
    
    # 4. Create FiftyOne Dataset
    dataset_name = "HAM10000_Model_Audit"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(name=dataset_name)
    
    # 5. Build Samples and Run Inference
    print(f"Running Inference & Building FiftyOne dataset with {len(image_paths)} samples...")
    samples = []
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            if i > 0 and i % 50 == 0:
                print(f"  Processed {i}/{len(image_paths)} images...")
                
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            row = df[df['image_id'] == img_id]
            
            sample = fo.Sample(filepath=img_path)
            
            if not row.empty:
                # Ground Truth
                true_label = str(row.iloc[0]['dx'])
                sample["ground_truth"] = fo.Classification(label=true_label)
                
                # Metadata
                sample["dx_type"] = str(row.iloc[0]['dx_type'])
                sample["sex"] = str(row.iloc[0]['sex'])
                sample["localization"] = str(row.iloc[0]['localization'])
                if pd.notna(row.iloc[0]['age']):
                    sample["age"] = float(row.iloc[0]['age'])
                    
                # Model Inference
                try:
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = transform(image).unsqueeze(0).to(device)
                    outputs = model(input_tensor)
                    _, preds_tensor = torch.max(outputs, 1)
                    pred_label = idx_to_class[preds_tensor.item()]
                    
                    # Store prediction in FiftyOne
                    sample["prediction"] = fo.Classification(label=pred_label)
                except Exception as e:
                    print(f"Error predicting {img_path}: {e}")
                    
            samples.append(sample)
        
    dataset.add_samples(samples)
    
    # 6. Evaluate Model (Computes False Positives, True Negatives, etc)
    print("\nEvaluating Predictions...")
    results = dataset.evaluate_classifications(
        "prediction",
        gt_field="ground_truth",
        eval_key="eval"
    )
    
    # Print Terminal Report
    print("\n" + "="*50)
    print("CLASSIFICATION EVALUATION REPORT")
    print("="*50)
    print(results.report())
    print("="*50 + "\n")
    
    # 7. Generate a Confusion Matrix Plot 
    # (By default, FiftyOne opens interactive plotly graphs in your browser)
    print("Generating Confusion Matrix Plot...")
    plot = results.plot_confusion_matrix()
    if plot:
        try:
            plot.show()
        except:
            pass

    # 8. Filter UI to show ONLY Incorrect Images
    # This prepares the specific "False Positive / False Negative" view you requested
    incorrect_view = dataset.match(
        fo.ViewField("prediction.label") != fo.ViewField("ground_truth.label")
    )
    
    print(f"Found {len(incorrect_view)} incorrectly identified images out of {len(image_paths)}.")
    print("Launching FiftyOne App filtered on INCORRECT PREDICTIONS... (Press Ctrl+C to close)")
    
    # Launch UI with the filtered list
    session = fo.launch_app(incorrect_view)
    session.wait()

if __name__ == "__main__":
    run_fiftyone_audit()
