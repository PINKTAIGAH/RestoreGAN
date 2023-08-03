from torch.utils.data import Dataset
import scipy.ndimage as ndimg
import torch
import config
import utils
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from kornia.geometry.transform import translate

class ImageGenerator(object):

    def __init__(self, psf, maxJitter, imageHeight):
        
        self.psf = psf
        self.ftPsf = torch.fft.fft2(self.psf)
        self.maxJitter = maxJitter
        self.imageHight = imageHeight

    def generateGroundTruth(self):

        whiteNoise = torch.randn(*self.ftPsf.shape)
        groundTruth = torch.fft.ifft2(self.ftPsf * torch.fft.fft2(whiteNoise))  
        return torch.real(groundTruth).type(torch.float32), whiteNoise.type(torch.float32)

    def generateShiftsHorizontal(self):
        shiftX = torch.randn((config.IMAGE_SIZE, 1))
        shiftY = torch.zeros_like(shiftX)
        return torch.cat([shiftX, shiftY], 1) * config.MAX_JITTER

    def generateShiftsVertical(self):
        shiftY = torch.randn((config.IMAGE_SIZE, 1))
        shiftX = torch.zeros_like(shiftY)
        return torch.cat([shiftX, shiftY], 1) * config.MAX_JITTER
    
    def shiftImageHorizontal(self, input, shifts, isBatch=False):
        if not isBatch:
            input = torch.unsqueeze(input, 0)
            shifts = torch.unsqueeze(shifts, 0)

        if len(input.shape) != 4:
            raise Exception("Input image must be of dimention 4: (B, C, H, W)")
        if len(shifts.shape) !=3:
            raise Exception("Shifts must be of the shape (B, H, 2)")

        B, _, H, _ = input.shape
        output = torch.zeros_like(input)
        for i in range(B):
            singleImage = torch.unsqueeze(torch.clone(input[i]),0)
            singleShift = torch.clone(shifts[i])
            for j in range(H):
                output[i, :, j, :] = translate(singleImage[:, :, j, :],
                                               torch.unsqueeze(singleShift[j], 0),
                                               padding_mode="reflection",
                                               align_corners=False)
        return output
    
    def shiftImageVertical(self, input, shifts, isBatch=True):
        if not isBatch:
            input = torch.unsqueeze(input, 0)
            shifts = torch.unsqueeze(shifts, 0)
        if len(input.shape) != 4:
            raise Exception("Input image must be of dimention 4: (B, C, H, W)")
        if len(shifts.shape) !=3:
            raise Exception("Shifts must be of the shape (B, H, 2)")

        B, _, _, W = input.shape
        output = torch.zeros_like(input)
        for i in range(B):
            singleImage = torch.unsqueeze(torch.clone(input[i]),0)
            singleShift = torch.clone(shifts[i])
            for j in range(W):
                output[i, :, :, j] = translate(singleImage[:, :, :, j],
                                               torch.unsqueeze(singleShift[j], 0),
                                               padding_mode="reflection",
                                               align_corners=True)
        return output

def test():
    pass 

if __name__ == "__main__":
    test()
