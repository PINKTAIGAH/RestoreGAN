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

        groundTruth, _ = self.filter.generateGroundTruth()
        groundTruth = torch.unsqueeze(groundTruth, 0)

        shifts = self.filter.generateShiftsHorizontal()
        # shiftsVertical = self.filter.generateShiftsVertical()
        
        shiftedImage = self.filter.newShiftImageHorizontal(groundTruth, shifts, 
                                                           isBatch=False,)
        # shiftedImageVertical = self.filter.newShiftImageVertical(groundTruth,
                                                      # shiftsVertical, isBatch=False)

        shiftedImage = torch.squeeze(shiftedImage, 0)
        # shiftedImageVertical = torch.squeeze(shiftedImageVertical, 0)

        shiftedImage = utils.normaliseTensor(shiftedImage)
        # shiftedImageVertical = utils.normaliseTensor(shiftedImageVertical)
        groundTruth = utils.normaliseTensor(groundTruth)

        shiftedImage = config.transforms(shiftedImage)
        groundTruth = config.transforms(groundTruth)
        # shiftedImageVertical = config.transforms(shiftedImageVertical)

        return shiftedImage, groundTruth, shifts

if __name__ == "__main__":

    N = 256 
    dataset = JitteredDataset(N, 20, 2)
    loader = DataLoader(dataset, batch_size=5)
    filter = ImageGenerator(config.PSF, config.MAX_JITTER, config.IMAGE_SIZE,)

    for x, y, shifts in loader:
        print(x.shape, y.shape, shifts.shape)
        z = filter.newShiftImageHorizontal(x, -shifts, isBatch=True)

        save_image(x, "images/Jittered.png")
        save_image(y, "images/Unjittered.png")
        save_image(z, "images/Dejittered.png")
