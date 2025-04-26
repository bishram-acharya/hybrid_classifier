import torch
import numpy as np
import cv2
from torchvision import transforms


class GradCAM:
    """
    Implements GradCAM visualization for HybridNet model.
    Focuses on the CNN backbone for visualization.
    """
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.model.eval()
        
        # Set target layer based on backbone architecture
        if target_layer_name is None:
            if model.backbone_type == 'efficientnet':
                # For EfficientNet, we target the last convolutional layer
                self.target_layer = model.cnn.features[-1]
            elif model.backbone_type == 'resnet':
                # For ResNet, we target the last residual block
                self.target_layer = model.cnn.layer4[-1]
            else:
                raise ValueError(f"Unsupported backbone type: {model.backbone_type}")
        else:
            # Allow specifying custom target layer
            self.target_layer = target_layer_name
            
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.register_hooks()
    
    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register forward and backward hooks
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """Remove hooks to prevent memory leaks"""
        self.forward_handle.remove()
        self.backward_handle.remove()
        
    def __del__(self):
        """Clean up hooks when object is deleted"""
        try:
            self.remove_hooks()
        except:
            pass
        
    def generate_cam(self, class_idx):
        """Generate class activation map for the specified class index"""
        # Get gradients and activations
        weights = torch.mean(self.gradients, dim=[2, 3])  # Global average pooling of gradients
        
        # Create weighted activation map
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32).to(self.activations.device)
        
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i, :, :]
            
        # Apply ReLU to focus on positive influences
        cam = torch.maximum(cam, torch.tensor(0, dtype=torch.float32).to(cam.device))
        
        # Normalize between 0-1
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
            
        return cam.detach().cpu().numpy()
    
    def visualize(self, img_tensor, class_idx, overlay_alpha=0.5):
        """
        Generate CAM visualization for a given image and class.
        
        Args:
            img_tensor (torch.Tensor): Image tensor [C, H, W]
            class_idx (int): Class index to visualize
            overlay_alpha (float): Transparency for heatmap overlay
            
        Returns:
            tuple: (original image, heatmap, overlaid image)
        """
        # Get original image
        orig_img = img_tensor.permute(1, 2, 0).cpu().numpy()
        orig_img = np.clip(orig_img, 0, 1)
        
        # Generate CAM
        cam = self.generate_cam(class_idx)
        
        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Overlay on original image
        overlaid = overlay_alpha * heatmap + (1 - overlay_alpha) * orig_img
        overlaid = np.clip(overlaid, 0, 1)
        
        return orig_img, heatmap, overlaid
    
    def process_example(self, img_tensor, pred_label, model, vit_processor, extract_lbp_fn, device):
        """
        Process a single example to generate GradCAM visualization
        
        Args:
            img_tensor: Single image tensor [C, H, W]
            pred_label: Predicted class index
            model: HybridNet model
            vit_processor: Vision transformer processor
            extract_lbp_fn: Function to extract LBP features
            device: Device to run computations on
            
        Returns:
            tuple: (original image, heatmap, overlay)
        """
        # Add batch dimension
        img_input = img_tensor.unsqueeze(0)
        
        # Get ViT inputs and LBP features for this sample
        img_pil = transforms.ToPILImage()(img_tensor.cpu())
        lbp_feat = torch.tensor([extract_lbp_fn(img_pil)], dtype=torch.float32).to(device)
        vit_inputs = vit_processor(images=[img_pil], return_tensors="pt")
        vit_inputs = {k: v.to(device) for k, v in vit_inputs.items()}
        
        # Reset gradients
        model.zero_grad()
        
        # Forward pass
        outputs = model(img_input, vit_inputs, lbp_feat)
        
        # Backward pass for predicted class
        outputs[0, pred_label].backward()
        
        # Generate GradCAM visualizations
        orig, heatmap, overlay = self.visualize(img_tensor, pred_label)
        
        return orig, heatmap, overlay