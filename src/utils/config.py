import torch

"""
Hyperparameter definiton
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Change to GPU only in future
TRAIN_DIR = None
VALIDATION_DIR = None
LEARNING_RATE = 1e-4
BATCH_SIZE = 64         # This is subject to change
N_WORKERS = 2
IMAGE_SIZE = 265        # Variable
IMAGE_CHANNELS = 1
FEATURES_DISCRIMINATOR = 16
FEATURES_GENERATOR = None       # Will have to investigate
CRITIC_ITERATIONS = None        # Will have to investigate
LAMBDA_GP = 10
N_EPOCH = 500                   # Will have to investigate
SAVE_MODEL = True
LOAD_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINR_GEN = "gen.pth.tar"
