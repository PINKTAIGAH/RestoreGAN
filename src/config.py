import numpy as np 
import torch
from skimage.filters import gaussian
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import transforms as transform

def normalise(x):
    if np.sum(x) == 0:
        raise Exception("Divided by zero. Attempted to normalise a zero tensor")

    return x/np.sum(x**2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#TRAIN_DIR = "/home/giorgio/Desktop/cell_dataset/train/"
TRAIN_DIR = "/home/brunicam/myscratch/p3_scratch/cell_dataset/train/"
#VAL_DIR = "/home/giorgio/Desktop/cell_dataset/val/"
VAL_DIR = "/home/brunicam/myscratch/p3_scratch/cell_dataset/val/"
LEARNING_RATE = 1e-4
SCHEDULAR_DECAY = 0.5
SCHEDULAR_PATIENCE = 20
BATCH_SIZE = 16
NUM_WORKERS = 2
MAX_JITTER = 4
IMAGE_SIZE = 128
IMAGE_JITTER = 5 
CHANNELS_IMG = 1 
CHANNELS_OUT = 1
SIGMA = 10 
LAMBDA_CONTENT = 1.0
LAMBDA_JITTER = 1.0
LAMBDA_GP = 10
SCALING_FACTOR = 6
NUM_EPOCHS = 500
LOAD_MODEL = False 
SAVE_MODEL = True
CHECKPOINT_DISC = "../models/disc.pth.tar"
CHECKPOINT_GEN = "../models/gen.pth.tar"
WRITER_REAL = SummaryWriter("../runs/real")
WRITER_FAKE = SummaryWriter("../runs/fake")

kernal = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
kernal[IMAGE_SIZE//2, IMAGE_SIZE//2] = 1
PSF = torch.from_numpy(normalise(gaussian(kernal, SIGMA)))

transforms = transform.Compose([
    transform.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)],   # generalise for multi channel
        [0.5 for _ in range(CHANNELS_IMG)],
    ),
])

transformsCell = transform.Compose([
    transform.ToTensor(),
    transform.Grayscale(),
])
