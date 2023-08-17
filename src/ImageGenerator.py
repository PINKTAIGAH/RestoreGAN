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
    """
    A class used to generate white noise images, generate shift flow maps and 
    to shift images according to a given flow map.

    Parameters
    ----------
    psf: torch.FloatTensor
        A 2D point spread function with equal hight and width dimention as the 
        imageHight parameter

    imageHight: int
        Hight of image. (Currently class assumes square images, hence imageHight
        also represents width of image)

    correlationLength: float
        Represents correlation length of individual instances of jitter in an image.
        Corresponds to standard deviation of gaussian envelopes in shift generation.

    paddingWidth: int
        The number of pixels that will be added as padding on each edges of the image.

    maxJitter: float
        The maximum (and minimum) value of pixel shift. 


    Atributes
    ---------
    ftPsf: torch.ConplexFloatTensor
        The fourier transform of the psf parameter

    pad: torchvision.transforms.transforms.Pad instances
        The Pad instance initialised with the paddingWidth parameter

    identityFlowMap: torch.FloatTensor
        Flow map where each vector represents the position of it's corresponding 
        pixel in the flow map vector space. Tensor shape is (H, W, 2) 
    """

    def __init__(self, psf, imageHeight, correlationLength, paddingWidth, maxJitter):
        
        self.psf = psf
        self.ftPsf = torch.fft.fft2(self.psf)
        self.imageHight = imageHeight
        self.correlationLength = correlationLength
        self.pad = Pad(paddingWidth)
        self.maxJitter = maxJitter

        # Using affine grid calculate identity flow map using identity matrix
        identifyMatrix = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        self.identityFlowMap = torch.squeeze(F.affine_grid(identifyMatrix,
                                             [1, 1, self.imageHight,
                                              self.imageHight]), 0) 

    def generateGroundTruth(self, padImage=True):
        """
        Return an image of white noise convolved with a point spread funtion.
        Shape of image is (1, H, W). 
        If padImage parameter is set to True, zero padding will be added to the
        image according to the paddingWidth parameter.

        Parameters
        ----------
        padImage: bool, optional
            Allow padding of generated image

        Returns
        -------
        groundTruth: torch.FloatTensor
            Tensor containing white noise image.
        """

        whiteNoise = torch.randn(*self.ftPsf.shape)
        # Convolve white noise with psf using convolution theorem
        groundTruth = torch.fft.ifft2(self.ftPsf * torch.fft.fft2(whiteNoise))  
        groundTruth = torch.unsqueeze(groundTruth, 0)
        
        if not padImage:
            # Return image without padding
            return torch.real(groundTruth)

        return self.pad(torch.real(groundTruth))

    def wavelet(self, x, x_0=0.0, std=1.0):
        """

        """
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
        step = self.identityFlowMap[0, 1, 0] - self.identityFlowMap[0, 0, 0]   
        
        flowMapShift, flowMapUnshift = (torch.clone(self.identityFlowMap),
                                        torch.clone(self.identityFlowMap))
        flowMapShift[:, :, 0] += shiftMap*step 
        flowMapUnshift[:, :, 0] -= shiftMap*step
        
        return flowMapShift, flowMapUnshift, shiftMap

    def shift(self, input, flowMap, isBatch=True):
        if not isBatch:
            input = torch.unsqueeze(input, 0)
            flowMap = torch.unsqueeze(flowMap, 0)

        assert len(input.shape) == 4 and len(flowMap.shape) == 4,\
                "Input image and flowMap must have shape 4"
        
        output =  F.grid_sample(input, flowMap, mode="bicubic", padding_mode="zeros",
                         align_corners=False)

        if not isBatch:
            return torch.squeeze(output, 0)

        return output

def test():

    filter = ImageGenerator(config.PSF, config.IMAGE_SIZE, config.CORRELATION_LENGTH,
                            config.PADDING_WIDTH, config.MAX_JITTER)

    t1 = time()
    groundTruth = filter.generateGroundTruth()
    flowMapShift, flowMapUnshift, shiftMap = filter.generateFlowMap()
    shifted = filter.shift(groundTruth, flowMapShift, isBatch=False)
    t2 = time()
    unshifted = filter.shift(shifted, flowMapUnshift, isBatch=False)
    t3 = time()

    x = np.arange(config.IMAGE_SIZE)
    fig, (ax1,ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.scatter(x, shiftMap[0])
    ax2.scatter(x, shiftMap[1])
    ax3.scatter(x, shiftMap[2])
    ax4.scatter(x, shiftMap[3])
    plt.show()

    fig, ((ax1,ax2),(ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(groundTruth[0], cmap="gray")
    ax2.imshow(shifted[0], cmap="gray")
    ax3.imshow(unshifted[0], cmap="gray")
    ax4.imshow(groundTruth[0] - unshifted[0], cmap="gray")
    plt.show()
    print(f"Time taken to generate ground truth and shift: {t2-t1} s")
    print(f"Time taken to unshft image: {t3-t2} s")

if __name__ == "__main__":
    test()
