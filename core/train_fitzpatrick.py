import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import sys

sys.path.append(r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models")
from model import get_resnet50_model

# 1. Dataset Class mapped to the new Folder Structure
class FitzpatrickDataset(Dataset):
    def __init__(self, data_dir, transform=None, limit=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Consistent mapping with HAM10000 tests
        label_map = {
            'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 
            'mel': 4, 'nv': 5, 'vasc': 6
        }
        
        print("Scanning Fitzpatrick Subset (Tones 1-6)...")
        # Load from the tone-split folders 1-6
        for tone_folder in range(1, 7):
            tone_path = os.path.join(data_dir, str(tone_folder))
            if not os.path.exists(tone_path): continue
            
            for file_name in os.listdir(tone_path):
                if not file_name.endswith('.jpg'): continue
                
                # Extract label from prefix (e.g. "mel_0a7d0...jpg")
                prefix = file_name.split('_')[0]
                if prefix in label_map:
                    self.image_paths.append(os.path.join(tone_path, file_name))
                    self.labels.append(label_map[prefix])

        if limit is not None:
            self.image_paths = self.image_paths[:limit]
            self.labels = self.labels[:limit]
            
        print(f"Loaded {len(self.image_paths)} images from all skin tones.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Bypassing the OpenCV health checks for clean baseline training
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

def train_fitzpatrick(epochs=5, batch_size=16, experimental_limit=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training Fitzpatrick Baseline on Device: {device} ---")
    
    # Checkpoint setup
    save_path = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\fitzpatrick_weights.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\fitz_ham10000_subset"
    train_dataset = FitzpatrickDataset(data_dir, transform=transform, limit=experimental_limit)
    
    if len(train_dataset) == 0:
        print("ERROR: No images found. Did you run the filter script?")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Load Fresh Model (using the 7 classes to directly compare models later)
    print("Loading Fresh ResNet50 Architecture...")
    model, _ = get_resnet50_model(num_classes=7)
    model = model.to(device)
    
    # Optimizer & Loss 
    # Notice we don't skew the weights as heavily as we did for the clinical dataset 
    # to avoid introducing a new bias against benign lesions in darker skin tones.
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    idx_to_class = {0:'akiec', 1:'bcc', 2:'bkl', 3:'df', 4:'mel', 5:'nv', 6:'vasc'}

    best_loss = float('inf')
    print("\nStarting Training Loop on Diverse Skin Tones...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # Fetch latest prediction for the UI
            _, preds = torch.max(outputs, 1)
            latest_pred = idx_to_class[preds[-1].item()]
            actual_label = idx_to_class[labels[-1].item()]
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'pred': latest_pred, 'true': actual_label})
            pbar.update(1)
            
        pbar.close()
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss:.4f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best diverse model to {save_path}")

    print("\nTraining Complete! You now have a standalone AI model trained exclusively on multiracial skin tones.")

if __name__ == "__main__":
    # We will train on the ~1000 images over 10 epochs
    train_fitzpatrick(epochs=10, batch_size=16)
