import torch
from torch.utils.data import Dataset
from ImageGenerator import ImageGenerator
from JitterFilter import JitterFilter

class JitteredDataset(Dataset):
    def __init__(self, N, maxJitter, psfSigma=3, length=100):
        self.N = N
        self.length = length
        self.psfSigma = psfSigma
        self.maxJitter = maxJitter
        self.Generator = ImageGenerator(self.N)
        self.Filter = JitterFilter()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        groundTruthNumpy = self.Generator.genericNoise(sigma=self.psfSigma)
        jitteredTruthNumpy = self.Filter.rowJitter(groundTruthNumpy, self.N,
                                                   self.maxJitter)

        groundTruthTorch = torch.tensor(groundTruthNumpy, dtype=torch.float32) 
        jitteredTruthTorch = torch.tensor(jitteredTruthNumpy, dtype=torch.float32) 

        return groundTruthTorch, jitteredTruthTorch

if __name__ == "__main__":
    dataset = JitteredDataset(5, 2)
    jittered, truth = dataset[0] 
    print(jittered)
    
