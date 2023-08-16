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
        print(f"ground truth: {groundTruth.shape} \n flowMapShift: {flowMapShift.shape}")
        shifted = self.filter.shift(groundTruth, flowMapShift)

        groundTruth = torch.unsqueeze(groundTruth, 0)
        shifted = torch.squeeze(shifted, 0)
        flowMapUnshift = torch.squeeze(flowMapUnshift, 0)

        groundTruth = utils.normaliseTensor(groundTruth)
        shifted = utils.normaliseTensor(shifted)

        groundTruth = config.transforms(groundTruth)
        shifted = config.transforms(shifted)

        return shifted, groundTruth, flowMapUnshift

if __name__ == "__main__":

    N = 256
    dataset = JitteredDataset(N, 2000, )
    loader = DataLoader(dataset, batch_size=5)

    for i, images in enumerate(loader):
        x, y, unshiftMap = images
        

        if i == 0:
            save_image(x, "images/Jittered.png")
            save_image(y, "images/Unjittered.png")
            print(f"Jittered Shape ==> {x.shape} \n Ground truth Shape ==> {y.shape} \
            \nUnshift map Shape ==> {unshiftMap.shape}")
            print("First batch created")

