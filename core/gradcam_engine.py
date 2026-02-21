import cv2
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_cam(model, target_layer, input_tensor, original_image_bgr, target_category=None):
    """
    Generate a Grad-CAM visualization for a given image tensor.
    
    Args:
        model: PyTorch model.
        target_layer: The target layer to compute Grad-CAM for (e.g., model.layer4[-1]).
        input_tensor: 1xCxHxW tensor representing the preprocessed image.
        original_image_bgr: The original image in BGR format (H, W, 3) as a numpy array. 
                            Values MUST be floats in the range [0.0, 1.0].
        target_category: Integer index of the class to visualize. If None, uses the highest scoring class.
        
    Returns:
        A BGR numpy image array with the heatmap overlaid on the original image.
    """
    # Initialize the GradCAM object
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Specify the target category
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    
    # Generate the heatmap Grayscale array
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    # Extract the first image's heatmap from the batch
    grayscale_cam = grayscale_cam[0, :]
    
    # Make sure the original image is scaled between 0 and 1
    if original_image_bgr.dtype == np.uint8:
        original_image_bgr = original_image_bgr.astype(np.float32) / 255.0
        
    # Convert original BGR to RGB for the show_cam_on_image utility
    original_image_rgb = cv2.cvtColor(np.float32(original_image_bgr), cv2.COLOR_BGR2RGB)
    
    # Overlay the heatmap
    visualization = show_cam_on_image(original_image_rgb, grayscale_cam, use_rgb=True)
    
    # Convert back to BGR for OpenCV display/saving
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    
    return visualization_bgr
