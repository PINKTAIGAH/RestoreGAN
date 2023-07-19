import numpy as np
import torch 
from skimage.filters import gaussian
from scipy.signal import convolve2d


class ImageGenerator(object):
    def __init__(self, N=64):
        self.N = N

    def createKernalPSE(self):
        self.kernal = np.zeros((self.kernalSize, self.kernalSize))
        self.kernal[self.kernalSize//2, self.kernalSize//2] = 1
        self.kernal = gaussian(self.kernal, sigma=self.sigma)

    def genericNoise(self, nConvolutions=3, kernalSize=4, sigma=3):
        self.sigma = sigma
        self.kernalSize = kernalSize
        self.createKernalPSE()

        array = np.random.random((self.N, self.N))

        for _ in range(nConvolutions):
            array = convolve2d(array, self.kernal, boundary="wrap")
        return array

if __name__ == "__main__":
    Generator = ImageGenerator()
    print(Generator.genericNoise())

