"""
Local Binary Pattern feature extraction utility.
"""
import numpy as np
from skimage.feature import local_binary_pattern
from torchvision import transforms
import torch

def extract_lbp_features(img_pil, P=8, R=1):
    """
    Extract Local Binary Pattern features from PIL image.
    
    Args:
        img_pil (PIL.Image): Input image in PIL format
        P (int): Number of circularly symmetric neighbor set points
        R (int): Radius of circle
        
    Returns:
        numpy.ndarray: LBP histogram features
    """
    gray = np.array(transforms.Grayscale()(img_pil))
    lbp = local_binary_pattern(gray, P=P, R=R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), 
                          range=(0, P + 2), density=True)
    return hist.astype(np.float32)
