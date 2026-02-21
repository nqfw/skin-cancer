import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet50_model(num_classes=None):
    """
    Returns a PyTorch ResNet50 model initialized with the highly-accurate 
    IMAGENET1K_V2 tuned weights.
    
    If num_classes is provided, Replaces the final fully connected layer 
    for custom fine-tuning.
    """
    # Load the base model with the requested V2 weights
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    
    # If the user wants to finetune to a specific number of classes (e.g., 2 for Malignant/Benign)
    if num_classes is not None:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    return model, weights
