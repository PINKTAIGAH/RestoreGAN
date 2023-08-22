from skimage.filters import gaussian
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import transforms as transform

def _normalise(x):
    """
    Normalise a ndArray

    Parameters
    ----------
    x: ndArray
        Input array containing dataset to be normalised

    Returns
    -------
    output: ndArray
        Normalised array
    """
    if np.sum(x) == 0:
        raise Exception("Divided by zero. Attempted to normalise a zero tensor")

    return x/np.sum(x**2)

def _generatePointFunction():
    """
    Generate a image array of a point function with size of defined in config 
    file (excluding padding)
    
    Returns
    -------
    pointFunction: ndArray
        2D image array containing a point function
    """

    pointFunction = np.zeros((NOISE_SIZE, NOISE_SIZE))
    pointFunction[NOISE_SIZE//2, NOISE_SIZE//2] = 1
    return pointFunction

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
SCHEDULAR_DECAY = 0.2
SCHEDULAR_STEP = 20                         # Step size of learning rate schedular
OPTIMISER_WEIGHTS = (0.5, 0.999)            # Beta parameters of Adam optimiser
DISCRIMINATOR_ITERATIONS = 1                # Number of iteration disc is trained per gen training
NUM_WORKERS = 2
MAX_JITTER = 2
PADDING_WIDTH = 15
IMAGE_SIZE = 128 
NOISE_SIZE = IMAGE_SIZE - PADDING_WIDTH*2
SIGMA = 10                                  # Standard deviation of gaussian kernal for PSF
CHANNELS_IMG = 1                            # Colour channels of input image tensors 
CHANNELS_OUT = 2
CORRELATION_LENGTH = 10
NUM_EPOCHS =  1500
LAMBDA_CONTENT = 100
LAMBDA_JITTER =  100
LAMBDA_GP = 10
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_DISC = "../models/disc.pth.tar"
CHECKPOINT_GEN = "../models/gen.pth.tar"
MODEL_LOSSES_FILE = "../raw_data/model_losses.txt"
MODEL_LOSSES_TITLES = ["epoch", "disc_loss", "gen_loss"]

# WRITER_REAL = SummaryWriter("../runs/real")
# WRITER_FAKE = SummaryWriter("../runs/fake")
# WRITER_REAL = SummaryWriter("/home/brunicam/myscratch/p3_scratch/runs/real")
# WRITER_FAKE = SummaryWriter("/home/brunicam/myscratch/p3_scratch/runs/fake")

# Generate a point spread finction by applying a gaussian blur to a point function
pointFunction = _generatePointFunction()
PSF = torch.from_numpy(_normalise(gaussian(pointFunction, SIGMA))).type(torch.float32)

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
