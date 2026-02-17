import os

# WSL File Paths
RAW_TRAIN_DIR = '/home/vistaar/geotiffs/tier1'
RAW_TEST_DIR = '/home/vistaar/geotiffs/test'

# Separate Processed Data Folders
PROCESSED_TRAIN_DIR = '/home/vistaar/geotiffs/processed/train'
PROCESSED_TEST_DIR = '/home/vistaar/geotiffs/processed/test'
CHECKPOINT_DIR = '/home/vistaar/geotiffs/checkpoints'

os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters tailored for 8GB VRAM
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
TILE_SIZE = 256
NUM_CLASSES = 5 # 0: Background, 1: No Damage, 2: Minor, 3: Major, 4: Destroyed

# Loss Weights
LAMBDA_FOCAL_DICE = 1.0
LAMBDA_ORDINAL = 0.5
LAMBDA_BOUNDARY = 0.75

# CutMix Probability
CUTMIX_PROB = 0.5