from PIL import Image
import os
import torch
import utils
import config
from torch.utils.data import Dataset
from ImageGenerator import ImageGenerator
import matplotlib.pyplot as plt

class FileDataset(Dataset):
    """
    A instance of a torch.utils.Data.DataLoader class which will return a png image  
    saved at a specified directory along side a shifted version of the image when
    indexed and a flow map that will unshift the shifted image.

    Atributes
    ---------
    filter: torch.utils.Data.Dataloader
        An instance of the ImageGenerator class using the hyperparameters 
        defined in the config file.

    transform: torchvision.transforms.transforms.Compose instance
        An instance of the Compose class containing all the transforms that will
        be applied to the imported image sequentially

    maxJitter: float
        The maximum (and minimum) value of pixel shift. 

    listFiles: list of strings
        List containing the file name of every image located in the root 
        directory

    Parameters
    ----------
    rootDirectory: string
        Root directory containing the images to be imported.

    imageSize: int
        Hight and Width of final image tensor. Images from root directory will be
        randomlly cropped to the specified size.
    """
    def __init__(self, rootDirectory, imageSize,):
        self.rootDirectory = rootDirectory
        self.listFiles = os.listdir(self.rootDirectory)
        self.N = imageSize 
        self.maxJitter = config.MAX_JITTER 

        self.filter = ImageGenerator(config.IMAGE_SIZE, 
                                     config.CORRELATION_LENGTH, config.PADDING_WIDTH,
                                     config.MAX_JITTER)
        self.transform = config.transformsFile

    def __len__(self):
        return len(self.listFiles)

    def _getImage(self, index):
        """
        Retrieves indexed imagefrom root directory, applies transform attribute and
        returns image

        Parameters
        ----------
        index: int
            Index of image to be retrieved

        Returns
        -------
        image: torch.FloatTensor
            Returns indexed image with specified transformations applied
        """

        # Retrieve name of indexed image
        imageFile = self.listFiles[index]
        #Oobtain full directory of indexed image 
        imagePath = os.path.join(self.rootDirectory, imageFile)
        # Load indexed image as an instance of PIL.PngImagePlugin.PngImageFile 
        image = Image.open(imagePath)
        image = self.transform(image)
        return image

    def __getitem__(self, index):
        """
        Return a new ground truth, jittered image and unshift flow map

        Parameters
        ----------
        idx: int
            Index of dataset to be returned. Given generation method used, this
            parameter is unused

        Returns
        -------
        shifted: torch.FloatTensor
            Shifted image tensor

        groundTruth: torch.FloatTensor
            Unshifted version of image tensor

        unshiftMap: torch.FloatTensor
            Flow map tat will unshift shifted version of image to recuperate
            ground truth

        Notes
        -----
        Method used to calculate unshift flow map is not perfect and there is a
        non-neglegable difference between unshifted image using the unshiftMap vs
        the ground truth.
        
        Possible method to invert shift flow map is described here 
        (DOI:10.1007/978-3-642-38628-2_46)
        """
        groundTruth = self._getImage(index)

        flowMapShift, flowMapUnshift, _ = self.filter.generateFlowMap()
        shifted = self.filter.shift(groundTruth, flowMapShift, isBatch=False,)

        # Change range of image tensor to [0, 1]
        groundTruth = utils.normaliseTensor(groundTruth)
        shifted = utils.normaliseTensor(shifted)

        # Normalise tensors with gaussian distrubition with mean and std of 0.5
        groundTruth = config.transforms(groundTruth)
        shifted = config.transforms(shifted)

        return shifted, groundTruth, flowMapUnshift

def test():
    dataset = FileDataset(config.TRAIN_DIR, config.IMAGE_SIZE,)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(len(dataset)):
        x, y, _ = dataset[i]
        if i == 0:
            ax1.imshow(x[0], animated=True)
            ax2.imshow(y[0], animated=True)
        else:
            plt.cla()
            ax1.imshow(x[0], animated=True)
            ax2.imshow(y[0], animated=True)
            plt.draw()
            plt.pause(1)
        print(i)
    

if __name__ == "__main__":
    test()
