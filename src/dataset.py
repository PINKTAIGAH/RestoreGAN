import torch
import utils
import config
from torch.utils.data import Dataset, DataLoader
from ImageGenerator import ImageGenerator
from torchvision.utils import save_image

class JitteredDataset(Dataset):
    def __init__(self, imageSize, length):
        self.N = imageSize 
        self.length = length
        self.filter = ImageGenerator(config.PSF, config.IMAGE_SIZE,
                            config.CORRELATION_LENGTH, config.PADDING_WIDTH,
                                     config.MAX_JITTER)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        groundTruth = self.filter.generateGroundTruth()
        flowMapShift, flowMapUnshift, _ = self.filter.generateFlowMap()
        shifted = self.filter.shift(groundTruth, flowMapShift, isBatch=False)

        groundTruth = utils.normaliseTensor(groundTruth)
        shifted = utils.normaliseTensor(shifted)

        groundTruth = config.transforms(groundTruth)
        shifted = config.transforms(shifted)

        return shifted, groundTruth, flowMapUnshift

if __name__ == "__main__":

    N = 256
    filter = ImageGenerator(config.PSF, config.IMAGE_SIZE, config.CORRELATION_LENGTH,
                            config.PADDING_WIDTH, config.MAX_JITTER)
    dataset = JitteredDataset(N, 2000, )
    loader = DataLoader(dataset, batch_size=5)

    for i, images in enumerate(loader):
        x, y, unshiftMap = images
        deshifted = filter.shift(x, unshiftMap, isBatch=True) 
        
        if i == 0:
            save_image(x, "images/Jittered.png")
            save_image(y, "images/Unjittered.png")
            save_image(deshifted, "images/Deshifted.png")
            print(f"Jittered Shape ==> {x.shape} \n Ground truth Shape ==> {y.shape} \
            \nUnshift map Shape ==> {unshiftMap.shape}")
            print("First batch created")

