import utils
import config
from torch.utils.data import Dataset, DataLoader
from ImageGenerator import ImageGenerator
from torchvision.utils import save_image

class JitteredDataset(Dataset):
    """
    A instance of a torch.utils.Data.DataLoader class which will return an image  
    of white nouse convoluted with a point spread function allong side a shifted
    version of the image when indexed and a flow map that will unshift the shifted
    image.

    Atributes
    ---------
    filter: torch.utils.Data.Dataloader
        An instance of the ImageGenerator class using the hyperparameters 
        defined in the config file.

    Parameters
    ----------
    length: int, optional
        Length of dataset. Given the method used in the __getitem__ method, this
        parameter is arbitrary. (Yet nescesary for class to function properly)
    """
    def __init__(self, length=1):
        self.length = length
        self.filter = ImageGenerator(config.PSF, config.IMAGE_SIZE,
                            config.CORRELATION_LENGTH, config.PADDING_WIDTH,
                                     config.MAX_JITTER)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
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

        groundTruth = self.filter.generateGroundTruth()
        flowMapShift, flowMapUnshift, _ = self.filter.generateFlowMap()
        shifted = self.filter.shift(groundTruth, flowMapShift, isBatch=False)
 
        # Change range of image tensor to [0, 1]
        groundTruth = utils.normaliseTensor(groundTruth)
        shifted = utils.normaliseTensor(shifted)

        # Normalise tensors with gaussian distrubition with mean and std of 0.5
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
