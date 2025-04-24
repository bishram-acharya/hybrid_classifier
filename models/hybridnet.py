"""
HybridNet model combining CNN, Vision Transformer and LBP features.
"""
import torch
import torch.nn as nn
from torchvision import transforms 
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import ViTModel, ViTImageProcessor
from ..config import LBP_DIM, VIT_DIM, VIT_MODEL, DEVICE

class HybridNet(nn.Module):
    """
    Hybrid classification model that combines EfficientNet features,
    Vision Transformer features, and LBP texture features.
    """
    def __init__(self, num_classes, lbp_dim=LBP_DIM, vit_dim=VIT_DIM):
        super().__init__()
        self.effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.effnet.classifier = nn.Identity()
        
        self.vit = ViTModel.from_pretrained(VIT_MODEL)
        for param in self.vit.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(1280 + vit_dim + lbp_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, img_tensor, vit_inputs, lbp_feat):
        """
        Forward pass through the hybrid model.
        
        Args:
            img_tensor (torch.Tensor): Image tensor for EfficientNet
            vit_inputs (dict): Inputs for Vision Transformer
            lbp_feat (torch.Tensor): LBP texture features
            
        Returns:
            torch.Tensor: Logits for classification
        """
        eff_feat = self.effnet(img_tensor)
        with torch.no_grad():
            vit_out = self.vit(**vit_inputs).last_hidden_state[:, 0, :]  # CLS token (B, 768)
        vit_out_flat = vit_out.repeat(1, 197)
        combined = torch.cat([eff_feat, vit_out_flat, lbp_feat], dim=1)
        return self.classifier(combined)

def get_vit_processor():
    """
    Get the Vision Transformer image processor.
    
    Returns:
        ViTImageProcessor: The image processor for ViT
    """
    return ViTImageProcessor.from_pretrained(VIT_MODEL)

def prepare_batch(batch, vit_processor, extract_lbp_fn):
    """
    Prepare a batch for the hybrid model.
    
    Args:
        batch (tuple): (images, labels) batch from dataloader
        vit_processor (ViTImageProcessor): Image processor for ViT
        extract_lbp_fn (callable): Function to extract LBP features
        
    Returns:
        tuple: (image_tensors, vit_inputs, lbp_features, labels) on device
    """
    img_tensors, labels = batch
    batch_size = img_tensors.size(0)
    img_pils = [transforms.ToPILImage()(img_tensors[i].cpu()) for i in range(batch_size)]
    lbp_feats = torch.tensor(
        [extract_lbp_fn(pil) for pil in img_pils], 
        dtype=torch.float32
    ).to(DEVICE)
    
    vit_inputs = vit_processor(images=img_pils, return_tensors="pt")
    vit_inputs = {k: v.to(DEVICE) for k, v in vit_inputs.items()}
    
    return img_tensors.to(DEVICE), vit_inputs, lbp_feats, labels.to(DEVICE)