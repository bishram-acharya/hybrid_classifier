"""
Configuration settings for the hybrid classifier .
"""
import torch
import os

# Dataset paths
TRAIN_DIR = '/kaggle/input/breakhis-400x/BreaKHis 400X/train/'
TEST_DIR = '/kaggle/input/breakhis-400x/BreaKHis 400X/test/'

# Default Training parameters
BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS = 6
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.2

# Model parameters
LBP_DIM = 10
VIT_DIM = 768
VIT_MODEL = "google/vit-base-patch16-224-in21k"

# Model selection - choose between 'efficientnet' and 'resnet' . Default efficient net
BACKBONE_MODEL = 'efficientnet'

# Seeds for reproducibility
SEED = 42

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")