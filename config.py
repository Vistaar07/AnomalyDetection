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
BATCH_SIZE = 16
EPOCHS = 120          # Extended: cosine schedule spans full budget, no early collapse
LEARNING_RATE = 5e-5
TILE_SIZE = 512
NUM_CLASSES = 5       # 0: Background, 1: No Damage, 2: Minor, 3: Major, 4: Destroyed

# [Background, No-Damage, Minor, Major, Destroyed]
CLASS_WEIGHTS = [0.1, 1.0, 10.0, 3.5, 5.0]

# Loss Weights
LAMBDA_FOCAL = 2.0
LAMBDA_DICE = 1.5     # Increased: Dice directly optimises per-class overlap
LAMBDA_BOUNDARY = 0.5 # Reduced: boundary supervision less critical than getting damage classification right

# CutMix Probability
CUTMIX_PROB = 0.5