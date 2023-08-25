from skimage.filters import gaussian
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import transforms as transform

"""
Hyper parameters
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Directory of files containing image datasets
TRAIN_DIR = "/home/giorgio/Desktop/p06_images/train/"
# TRAIN_DIR = "/home/brunicam/myscratch/p3_scratch/cell_dataset/train/"
VAL_DIR = "/home/giorgio/Desktop/p06_images/val/"
# VAL_DIR = "/home/brunicam/myscratch/p3_scratch/cell_dataset/val/"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
SCHEDULAR_DECAY = 0.9
SCHEDULAR_STEP = 30                         # Step size of learning rate schedular
OPTIMISER_WEIGHTS = (0.5, 0.999)            # Beta parameters of Adam optimiser
DISCRIMINATOR_ITERATIONS = 4                # Number of iteration disc is trained per gen training
NUM_WORKERS = 2
MAX_JITTER = 2
PADDING_WIDTH = 15
IMAGE_SIZE = 128 
NOISE_SIZE = IMAGE_SIZE - PADDING_WIDTH*2
SIGMA = 10                                  # Standard deviation of gaussian kernal for PSF
CHANNELS_IMG = 1                            # Colour channels of input image tensors 
CHANNELS_OUT = 1
CORRELATION_LENGTH = 10
NUM_EPOCHS = 300
LAMBDA_CONTENT = 200
LAMBDA_JITTER =  100
LAMBDA_GP = 10
LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_DISC_LOAD = "../models/disc.pth.tar"
CHECKPOINT_GEN_LOAD = "../models/gen.pth.tar"

CHECKPOINT_DISC_SAVE = "../models/disc.pth.tar"
CHECKPOINT_GEN_SAVE = "../models/gen.pth.tar"

MODEL_LOSSES_FILE = "../raw_data/model_losses.txt"
MODEL_LOSSES_TITLES = ["epoch", "disc_loss", "gen_loss"]
TRAIN_IMAGE_FILE= "../evaluation/default"
EVALUATION_IMAGE_FILE = "../evaluation/metric"

# Evaluation hyperparameters
EVALUATION_EPOCHS = 50
EVALUATION_METRIC_FILE = "../raw_data/lambda.txt"

# WRITER_REAL = SummaryWriter("../runs/real")
# WRITER_FAKE = SummaryWriter("../runs/fake")
# WRITER_REAL = SummaryWriter("/home/brunicam/myscratch/p3_scratch/runs/real")
# WRITER_FAKE = SummaryWriter("/home/brunicam/myscratch/p3_scratch/runs/fake")

"""
Tensor transforms
"""

transforms = transform.Compose([
    transform.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)],   # generalise for multi channel
        [0.5 for _ in range(CHANNELS_IMG)],
    ),
])

transformsFile = transform.Compose([
    transform.ToTensor(),
    transform.RandomCrop(IMAGE_SIZE),
    transform.Grayscale(),
])

"""
Hyperparameter overwriting for automatic bash scripts 
"""
