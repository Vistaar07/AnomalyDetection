import os

# File Paths
RAW_TRAIN_DIR = "C:/geotiffs/tier1"
RAW_TEST_DIR = "C:/geotiffs/test"

# Separate Processed Data Folders
PROCESSED_TRAIN_DIR = "C:/geotiffs/processed/train"
PROCESSED_TEST_DIR = "C:/geotiffs/processed/test"
CHECKPOINT_DIR = "C:/geotiffs/checkpoints"

os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4 # Reduced from 2e-4 for the B4 encoder
TILE_SIZE = 512 # Changed from 256 to 512
NUM_CLASSES = 5 # 0: Background, 1: No Damage, 2: Minor, 3: Major, 4: Destroyed

# [Background, No-Damage, Minor, Major, Destroyed]
CLASS_WEIGHTS = [0.5, 1.0, 4.0, 2.0, 2.0]

# Loss Weights
LAMBDA_FOCAL = 2.0
LAMBDA_DICE = 1.0
LAMBDA_ORDINAL = 2.0
LAMBDA_BOUNDARY = 0.75

# CutMix Probability
CUTMIX_PROB = 0.0