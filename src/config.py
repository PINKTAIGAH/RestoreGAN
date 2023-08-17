from skimage.filters import gaussian
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import transforms as transform

def _normalise(x):
    if np.sum(x) == 0:
        raise Exception("Divided by zero. Attempted to normalise a zero tensor")

    return x/np.sum(x**2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "/home/giorgio/Desktop/p06_images/train/"
# TRAIN_DIR = "/home/brunicam/myscratch/p3_scratch/cell_dataset/train/"
VAL_DIR = "/home/giorgio/Desktop/p06_images/val/"
# VAL_DIR = "/home/brunicam/myscratch/p3_scratch/cell_dataset/val/"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SCHEDULAR_DECAY = 0.5
SCHEDULAR_PATIENCE = 20
NUM_WORKERS = 2
MAX_JITTER = 4
PADDING_WIDTH = 30
IMAGE_SIZE = 128 
NOISE_SIZE = IMAGE_SIZE - PADDING_WIDTH*2
SIGMA = 20
CHANNELS_IMG = 1 
CHANNELS_OUT = 2
CORRELATION_LENGTH = 10
NUM_EPOCHS =  1000
LAMBDA_CONTENT = 1.0
LAMBDA_JITTER = 1.0
LAMBDA_GP = 10
SCALING_FACTOR = 6
LOAD_MODEL = False 
SAVE_MODEL = True
CHECKPOINT_DISC = "../models/disc.pth.tar"
CHECKPOINT_GEN = "../models/gen.pth.tar"
WRITER_REAL = SummaryWriter("../runs/real")
WRITER_FAKE = SummaryWriter("../runs/fake")
# WRITER_REAL = SummaryWriter("/home/brunicam/myscratch/p3_scratch/runs/real")
# WRITER_FAKE = SummaryWriter("/home/brunicam/myscratch/p3_scratch/runs/fake")

kernal = np.zeros((NOISE_SIZE, NOISE_SIZE))
kernal[NOISE_SIZE//2, NOISE_SIZE//2] = 1
PSF = torch.from_numpy(_normalise(gaussian(kernal, SIGMA))).type(torch.float32)

transforms = transform.Compose([
    transform.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)],   # generalise for multi channel
        [0.5 for _ in range(CHANNELS_IMG)],
    ),
])

transformsCell = transform.Compose([
    transform.ToTensor(),
    transform.RandomCrop(IMAGE_SIZE),
    transform.Grayscale(),
])
