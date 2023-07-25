import sys
import torch
import utils
import config
from torch.utils.data import Dataset, DataLoader
from ImageGenerator import ImageGenerator
from JitterFilter import JitterFilter
from torchvision.utils import save_image
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter

class JitteredDataset(Dataset):
    def __init__(self, N, maxJitter, psfSigma=3, length=128, concatImages=False):
        self.N = N
        self.length = length
        self.maxJitter = maxJitter
        self.Generator = ImageGenerator(self.N)
        self.Filter = JitterFilter()
        self.concatImages = concatImages
        self.psfSigma = psfSigma

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        groundTruthNumpy = self.Generator.genericNoise(sigma=self.psfSigma,
                                                       kernalSize=65)
        jitteredTruthNumpy = self.Filter.rowJitter(groundTruthNumpy, self.N,
                                                   self.maxJitter,)

        # Image.fromarray(groundTruthNumpy*255).convert("RGB").save("x_np.png")
        # Image.fromarray(jitteredTruthNumpy*255).convert("RGB").save("y_np.png")

        groundTruthTorch = torch.tensor(groundTruthNumpy, dtype=torch.float32) 
        jitteredTruthTorch = torch.tensor(jitteredTruthNumpy, dtype=torch.float32) 

        groundTruthTorch = torch.unsqueeze(groundTruthTorch, 0)
        jitteredTruthTorch = torch.unsqueeze(jitteredTruthTorch, 0)

        if self.concatImages:
            return utils.tensorConcatinate(jitteredTruthTorch, groundTruthTorch)

        jitteredTruthTorch = utils.rescaleTensor(jitteredTruthTorch)
        groundTruthTorch = utils.rescaleTensor(groundTruthTorch)

        jitteredTruthTorch = config.transforms(jitteredTruthTorch)
        groundTruthTorch = config.transforms(groundTruthTorch)

        return jitteredTruthTorch, groundTruthTorch

if __name__ == "__main__":
    writerJittered = SummaryWriter("test/jittered")
    writerUnjittered = SummaryWriter("test/unjittered")

    N=256 
    dataset = JitteredDataset(N, 5)
    loader = DataLoader(dataset, batch_size=2)
    # sys.exit()
    for x, y in loader:
        
        imgGridJittered = torchvision.utils.make_grid(x, normalize=True)
        imgGridUnjittered = torchvision.utils.make_grid(y, normalize=True)
        
        writerJittered.add_image("Jittered", imgGridJittered)
        writerUnjittered.add_image("Unjittered", imgGridUnjittered)

        x=x*0.5+0.5
        y=y*0.5+0.5

        save_image(x, "images/Jittered.tiff")
        save_image(y, "images/Unjittered.tiff")

        sys.exit()
