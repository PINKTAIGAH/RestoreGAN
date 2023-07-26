import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 128
IMAGE_JITTER = 3
CHANNELS_IMG = 1 
CHANNELS_OUT = 1
LAMBDA_CONTENT = 1
LAMBDA_JITTER = 1
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False 
SAVE_MODEL = True
CHECKPOINT_DISC = "../models/disc.pth.tar"
CHECKPOINT_GEN = "../models/gen.pth.tar"
WRITER_REAL = SummaryWriter("../runs/real")
WRITER_FAKE = SummaryWriter("../runs/fake")
SOBEL_KERNAL = torch.tensor(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32
)

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transforms = transforms.Compose([
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)],   # generalise for multi channel
        [0.5 for _ in range(CHANNELS_IMG)],
    ),
])   
