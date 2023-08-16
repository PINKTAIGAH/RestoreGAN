from torch.utils.data import Dataset
from scipy.ndimage import shift as shiftImage
import torch
import config
import utils
import numpy as np
from time import time
import matplotlib.pyplot as plt
from torchvision.transforms import Pad
import torch.nn.functional as F

class ImageGenerator(Dataset):

    def __init__(self, psf, imageHeight, correlationLength, paddingWidth, maxJitter):
        
        self.psf = psf
        self.ftPsf = torch.fft.fft2(self.psf)
        self.imageHight = imageHeight
        self.correlationLength = correlationLength
        self.pad = Pad(paddingWidth)
        self.maxJitter = maxJitter

        identifyAffine = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        self.identityFlowMap = F.affine_grid(identifyAffine,
                                             [1, 1, self.imageHight, self.imageHight]) 

    def generateGroundTruth(self):

        whiteNoise = torch.randn(*self.ftPsf.shape)
        groundTruth = torch.fft.ifft2(self.ftPsf * torch.fft.fft2(whiteNoise))  
        return self.pad(torch.real(groundTruth).type(torch.float32))

    def wavelet(self, x, x_0=0.0, std=1.0):
        return np.exp(-(x-x_0)**2/(2*std**2))
    
    def generateSignal(self, x, frequency):
        phase = np.random.uniform(0, 2*np.pi)
        return np.sin(2*np.pi*frequency*x + phase)

    def generateShiftMap(self):
        
        shiftMap = np.empty((self.imageHight, self.imageHight))
        waveletCentersDistance = np.random.normal(self.correlationLength*3,
                                                  self.correlationLength)
        waveletCenters = np.arange(0, self.imageHight, waveletCentersDistance)

        for i in range(self.imageHight):
            x = np.arange(self.imageHight)
            yFinal = np.zeros_like(x, dtype=np.float64)

            frequency = int(np.random.uniform(10, 100))
            waveletCentersDistance = np.random.normal(self.correlationLength*4.5,
                                                  self.correlationLength)
            waveletCenters = np.arange(0, self.imageHight, waveletCentersDistance)
            for _, val in enumerate(waveletCenters):
                jitter = np.random.uniform(0.5, self.maxJitter)
                y = self.generateSignal(x, frequency)
                yWavelet = self.wavelet(x, val, self.correlationLength)
                yFinal += utils.adjustArray(y * yWavelet)*jitter*2
            shiftMap[i] = yFinal
        return torch.from_numpy(shiftMap)

    def generateFlowMap(self,):
        shiftMap = self.generateShiftMap()
        step = self.identityFlowMap[0, 0, 1, 0] - self.identityFlowMap[0, 0, 0, 0]   
        
        flowMapShift, flowMapUnshift = (torch.clone(self.identityFlowMap),
                                        torch.clone(self.identityFlowMap))
        flowMapShift[:, :, :, 0] += torch.unsqueeze(shiftMap*step, 0) 
        flowMapUnshift[:, :, :, 0] -= torch.unsqueeze(shiftMap*step, 0)
        
        return flowMapShift, flowMapUnshift, shiftMap

    def shift(self, input, flowMap):
        input = torch.unsqueeze(torch.unsqueeze(input, 0) ,0)
        return F.grid_sample(input, flowMap, mode="bicubic", padding_mode="zeros",
                         align_corners=False)

def test():

    filter = ImageGenerator(config.PSF, config.IMAGE_SIZE, config.CORRELATION_LENGTH,
                            config.PADDING_WIDTH, config.MAX_JITTER)

    t1 = time()
    groundTruth = filter.generateGroundTruth()
    flowMapShift, flowMapUnshift, shiftMap = filter.generateFlowMap()
    shifted = torch.squeeze(filter.shift(groundTruth, flowMapShift), 0)
    t2 = time()
    unshifted = filter.shift(shifted[0], flowMapUnshift)
    t3 = time()

    x = np.arange(config.IMAGE_SIZE)
    fig, (ax1,ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.scatter(x, shiftMap[0])
    ax2.scatter(x, shiftMap[1])
    ax3.scatter(x, shiftMap[2])
    ax4.scatter(x, shiftMap[3])
    plt.show()

    fig, ((ax1,ax2),(ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(groundTruth, cmap="gray")
    ax2.imshow(shifted[0], cmap="gray")
    ax3.imshow(unshifted[0, 0], cmap="gray")
    ax4.imshow(groundTruth - unshifted[0, 0], cmap="gray")
    plt.show()
    print(f"Time taken to generate ground truth and shift: {t2-t1} s")
    print(f"Time taken to unshft image: {t3-t2} s")

if __name__ == "__main__":
    test()
