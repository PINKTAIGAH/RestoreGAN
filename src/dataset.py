import torch
import utils
import config
from torch.utils.data import Dataset, DataLoader
from ImageGenerator import ImageGenerator
from torchvision.utils import save_image

class JitteredDataset(Dataset):
    def __init__(self, imageSize, length, maxJitter,):
        self.N = imageSize 
        self.length = length
        self.maxJitter = maxJitter
        self.filter = ImageGenerator(config.PSF, config.MAX_JITTER, config.IMAGE_SIZE,)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        groundTruth, whiteNoise = self.filter.generateGroundTruth()
        shifts = self.filter.generateShifts()
        shiftsVertical = self.filter.generateShifts()
        shiftedImage = self.filter.shiftImage(groundTruth, shifts)
        shiftedImageVertical = self.filter.verticalShiftImage(shiftedImage,
                                                      shiftsVertical)

        groundTruth = torch.unsqueeze(groundTruth, 0)
        shiftedImage = torch.unsqueeze(shiftedImage, 0)
        shiftedImageVertical = torch.unsqueeze(shiftedImageVertical, 0)

        shiftedImage = utils.normaliseTensor(shiftedImage)
        shiftedImageVertical = utils.normaliseTensor(shiftedImageVertical)
        groundTruth = utils.normaliseTensor(groundTruth)

        shiftedImage = config.transforms(shiftedImage)
        groundTruth = config.transforms(groundTruth)
        shiftedImageVertical = config.transforms(shiftedImageVertical)

        return shiftedImageVertical, groundTruth

if __name__ == "__main__":

    N = 256 
    dataset = JitteredDataset(N, 20, 2)
    loader = DataLoader(dataset, batch_size=5)

    for x, y in loader:

        save_image(x, "images/Jittered.png")
        save_image(y, "images/Unjittered.png")
