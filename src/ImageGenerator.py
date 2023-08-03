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

    def generateShifts(self):
        return torch.randn(self.imageHight-1, dtype=torch.float32)*self.maxJitter

    
    def shiftImage(self, image, shifts):
        if len(image.shape) < 2:
            raise Exception("Can only tensors with a minimum of 2 dimentions") 

        totalShifts = torch.hstack([torch.tensor([0]), torch.cumsum(shifts, dim=0)])
        imageHight, imageWidth = image.shape[-2:]
        shiftedImage = torch.zeros_like(image)

        image = image.numpy() 
        shiftedImage = shiftedImage.numpy()
        totalShifts = totalShifts.numpy()

        for i, shift in enumerate(totalShifts):
            shiftedImage[i,:] = ndimg.shift(image[i,:], shift, output=None, order=3,
                                   cval=0.0, mode="wrap", prefilter=True)
        return torch.from_numpy(shiftedImage).type(torch.float32)

    def verticalShiftImage(self, image, shifts):
        if len(image.shape) < 2:
            raise Exception("Can only tensors with a minimum of 2 dimentions") 

        totalShifts = torch.hstack([torch.tensor([0]), torch.cumsum(shifts, dim=0)])
        imageHight, imageWidth = image.shape[-2:]
        shiftedImage = torch.zeros_like(image)

        image = image.numpy() 
        shiftedImage = shiftedImage.numpy()
        totalShifts = totalShifts.numpy()

        for i, shift in enumerate(totalShifts):
            shiftedImage[:,i] = ndimg.shift(image[:, i], shift, output=None, order=3,
                                   cval=0.0, mode="wrap", prefilter=True)
        return torch.from_numpy(shiftedImage).type(torch.float32)

    def unshiftImage(self, image, shifts):
        if len(image.shape) < 2:
            raise Exception("Can only tensors with a minimum of 2 dimentions") 

        totalShifts = torch.hstack([torch.tensor([0]), torch.cumsum(shifts, dim=0)])
        imageHight, imageWidth = image.shape[-2:]
        shiftedImage = torch.zeros_like(image)

        image = image.numpy() 
        shiftedImage = shiftedImage.numpy()
        totalShifts = totalShifts.numpy()

        for i, shift in enumerate(totalShifts):
            shiftedImage[i,:] = ndimg.shift(image[i,:], -shift, output=None, order=3,
                                   cval=0.0, mode="wrap", prefilter=True)
        return torch.from_numpy(shiftedImage).type(torch.float32)

    def newShiftImageHorizontal(self, input, shifts, isBatch=True):
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
                                               align_corners=True)
        return output
    
    def newShiftImageVertical(self, input, shifts):
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

    filter = ImageGenerator(config.PSF, config.MAX_JITTER, config.IMAGE_SIZE,)

    groundTruth, whiteNoise = filter.generateGroundTruth()
    shifts = filter.generateShifts()
    shiftsVertical = filter.generateShifts()
    shiftedImage = filter.shiftImage(groundTruth, shifts)
    shiftedImageVertical = filter.verticalShiftImage(shiftedImage,
            shiftsVertical)

    groundTruth = torch.unsqueeze(groundTruth, 0)
    whiteNoise = torch.unsqueeze(whiteNoise, 0)
    shiftedImage = torch.unsqueeze(shiftedImage, 0)
    shiftedImageVertical = torch.unsqueeze(shiftedImageVertical, 0)

    whiteNoise = utils.normaliseTensor(whiteNoise)
    shiftedImage = utils.normaliseTensor(shiftedImage)
    groundTruth = utils.normaliseTensor(groundTruth)
    shiftedImageVertical = utils.normaliseTensor(shiftedImageVertical)

    save_image(shiftedImage, "test.png", )
    print(shifts)

    fig, (ax1,ax2, ax3) = plt.subplots(1,3)
    ax3.imshow(shiftedImageVertical[0])
    ax2.imshow(shiftedImage[0])
    ax1.imshow(groundTruth[0])
    # print(whiteNoise.min().item(), whiteNoise.max().item())
    # print(groundTruth.min().item(), groundTruth.max().item())
    print(shifts.min().item(), shifts.max().item())
    plt.show()

    

if __name__ == "__main__":
    test()
