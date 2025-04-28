import torch
import numpy as np
import cv2
from torchvision import transforms
from collections import OrderedDict


class MultiLayerGradCAM:
    """
    Implements GradCAM visualization for multiple layers in a HybridNet model.
    Visualizes first layer, two intermediate layers, and final layer of the CNN backbone.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Target layers for different backbone architectures
        self.target_layers = self._get_target_layers()
            
        # Storage for activations and gradients
        self.activations = OrderedDict()
        self.gradients = OrderedDict()
        
        # Hooks for each layer
        self.hooks = []
        
        # Register hooks
        self.register_hooks()
    
    def _get_target_layers(self):
        """Identify target layers based on the backbone architecture"""
        target_layers = OrderedDict()
        
        if self.model.backbone_type == 'efficientnet':
            # For EfficientNet
            target_layers['first'] = self.model.cnn.features[0]  # First layer
            
            # Two intermediate layers
            n = len(self.model.cnn.features)
            target_layers['intermediate1'] = self.model.cnn.features[n//3]
            target_layers['intermediate2'] = self.model.cnn.features[2*n//3]
            
            # Final layer
            target_layers['final'] = self.model.cnn.features[-1]
            
        elif self.model.backbone_type == 'resnet':
            # For ResNet
            target_layers['first'] = self.model.cnn.conv1  # First layer
            
            # Two intermediate layers
            target_layers['intermediate1'] = self.model.cnn.layer2[-1]
            target_layers['intermediate2'] = self.model.cnn.layer3[-1]
            
            # Final layer
            target_layers['final'] = self.model.cnn.layer4[-1]
            
        else:
            raise ValueError(f"Unsupported backbone type: {self.model.backbone_type}")
            
        return target_layers
    
    def register_hooks(self):
        """Register forward and backward hooks for all target layers"""
        for name, layer in self.target_layers.items():
            # Create closure to capture the layer name
            def get_forward_hook(layer_name):
                def forward_hook(module, input, output):
                    self.activations[layer_name] = output
                return forward_hook
            
            def get_backward_hook(layer_name):
                def backward_hook(module, grad_input, grad_output):
                    self.gradients[layer_name] = grad_output[0]
                return backward_hook
            
            # Register hooks with the appropriate layer name
            forward_handle = layer.register_forward_hook(get_forward_hook(name))
            backward_handle = layer.register_full_backward_hook(get_backward_hook(name))
            
            # Store hooks for later removal
            self.hooks.append(forward_handle)
            self.hooks.append(backward_handle)
    
    def remove_hooks(self):
        """Remove hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def __del__(self):
        """Clean up hooks when object is deleted"""
        try:
            self.remove_hooks()
        except:
            pass
        
    def generate_cam(self, layer_name, class_idx):
        """Generate class activation map for the specified layer and class index"""
        # Get gradients and activations for the specified layer
        gradients = self.gradients[layer_name]
        activations = self.activations[layer_name]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3])  # weights shape: [batch_size, num_channels]
        
        # Create weighted activation map
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(activations.device)
        
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
            
        # Apply ReLU to focus on positive influences
        cam = torch.maximum(cam, torch.tensor(0, dtype=torch.float32).to(cam.device))
        
        # Normalize between 0-1
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
            
        return cam.detach().cpu().numpy()
    
    def visualize(self, img_tensor, class_idx, layer_name, overlay_alpha=0.5):
        """
        Generate CAM visualization for a given image, class, and layer.
        
        Args:
            img_tensor (torch.Tensor): Image tensor [C, H, W]
            class_idx (int): Class index to visualize
            layer_name (str): Name of the layer to visualize
            overlay_alpha (float): Transparency for heatmap overlay
            
        Returns:
            tuple: (original image, heatmap, overlaid image)
        """
        # Get original image
        orig_img = img_tensor.permute(1, 2, 0).cpu().numpy()
        orig_img = np.clip(orig_img, 0, 1)
        
        # Generate CAM for the specified layer
        cam = self.generate_cam(layer_name, class_idx)
        
        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Overlay on original image
        overlaid = overlay_alpha * heatmap + (1 - overlay_alpha) * orig_img
        overlaid = np.clip(overlaid, 0, 1)
        
        return orig_img, heatmap, overlaid
    
    def visualize_all_layers(self, img_tensor, class_idx, overlay_alpha=0.5):
        """
        Generate CAM visualizations for all target layers.
        
        Args:
            img_tensor (torch.Tensor): Image tensor [C, H, W]
            class_idx (int): Class index to visualize
            overlay_alpha (float): Transparency for heatmap overlay
            
        Returns:
            dict: Dictionary of (heatmap, overlaid image) tuples for each layer
        """
        results = {}
        orig_img = img_tensor.permute(1, 2, 0).cpu().numpy()
        orig_img = np.clip(orig_img, 0, 1)
        
        for layer_name in self.target_layers.keys():
            _, heatmap, overlay = self.visualize(img_tensor, class_idx, layer_name, overlay_alpha)
            results[layer_name] = (heatmap, overlay)
            
        return orig_img, results
    
    def process_example(self, img_tensor, pred_label, model, vit_processor, extract_lbp_fn, device):
        """
        Process a single example to generate GradCAM visualizations for all layers
        
        Args:
            img_tensor: Single image tensor [C, H, W]
            pred_label: Predicted class index
            model: HybridNet model
            vit_processor: Vision transformer processor
            extract_lbp_fn: Function to extract LBP features
            device: Device to run computations on
            
        Returns:
            tuple: (original image, results dictionary with layer visualizations)
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
        
        # Generate GradCAM visualizations for all layers
        orig_img, results = self.visualize_all_layers(img_tensor, pred_label)
        
        return orig_img, results

