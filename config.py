import os

# WSL File Paths
# config.py
RAW_DATA_DIR = r'\\wsl$\Ubuntu-22.04\home\vistaar\geotiffs\tier1'
PROCESSED_DATA_DIR = './data/processed'
CHECKPOINT_DIR = './checkpoints'

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters tailored for 8GB VRAM
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
TILE_SIZE = 256
NUM_CLASSES = 5 # 0: Background, 1: No Damage, 2: Minor, 3: Major, 4: Destroyed

# Loss Weights
LAMBDA_FOCAL_DICE = 1.0
LAMBDA_ORDINAL = 0.5
LAMBDA_BOUNDARY = 0.75

# CutMix Probability
CUTMIX_PROB = 0.5