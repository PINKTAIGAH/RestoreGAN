import numpy as np
import torch
import config
from torchvision.utils import save_image
import torch.nn.functional as F

def save_some_examples(gen, val_loader, epoch, folder, filter):
    """
    Save examples of output from generator as png images at a specified folder

    Parameters
    ----------
    gen: torch.nn.Module instance
        A generator neural network that will output image (or data required to 
        generate an output image)

    val_loader: torch.utils.Data.DataLoader instance
        A dataloader containing dataser that will be input in the generator
    
    epoch: int
        Epoch at which example is being taken

    folder: string
        Directory where output image will be saved

    filter: torch.utils.Data.Dataset instance
        Class constining method required to unshift image using generator's 
        outputted flowmap
    """
    # Unpack jittered (x) and ground truth (y) images from dataloader and send to device
    x, y, _ = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        # Generate flow map with generator and use it to unshift jittered image
        unshift_map_fake  = gen(x)
        y_fake = filter.shift(x, unshift_map_fake, isBatch=True)
        # Remove normalisation
        y_fake = y_fake * 0.5 + 0.5  
        # Save png of jittered, ground truth and generated unjitted image respectivly
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimiser, filename="my_checkpoint.pth.tar"):
    """
    Save dictionary of parameters of model and optimiser to specidied directory 
    in order to be loaded at a later time.

    Parameters
    ----------
    model: torch.nn.Module instance
        Neural network model to be saved

    optimiser: torch.optim instance
        Optimiser of model to be saved

    filename: string, optional
        Directory where model and optimiser will be saved
        
    """
    print("==> Saving checkpoint")
    # Dictionary constaining model and optimiser state parameters
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimiser.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimiser, lr):
    """
    Load previously saved model and optimisers by assingning saved dictionaries 
    containing state parameters to inputted model and optimiser.

    Parameters
    ----------
    checkpoint_file: string
        Directory of file containing state dictionaries of previously saved model
        and optimiser

    model: torch.nn.Module instance
        Neural network model where state dictionary will be loaded 

    optimiser: torch.optim instance
        Optimiser of model where state dictionary will be loaded 

    lr: torch.TensorFloat
        Value of learning rate that is currently being used to train model

    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # Load saved state dictionaries 
    model.load_state_dict(checkpoint["state_dict"])
    optimiser.load_state_dict(checkpoint["optimizer"])

    # Assign current learning rate to the optimiser
    for param_group in optimiser.param_groups:
        param_group["lr"] = lr

def _findMin(tensor):
    """
    Find minimum value of each batch of image tensor of shape (B, ..., H, W) and
    return tensor of same shape containing minimum value at each element for each
    batch

    Parameters
    ----------
    tensor: torch.FloatTensor
       Input tensor  

    Returns
    -------
    minTensor: torch.FloatTensor
        Tensor containing the minimum value of an image at each batch.
        Returns tensor of same shape as input

    Note
    ----
    Function assumes square images, hence H == W
    Function assumes only one colour channel
    """
    # Size of image
    N = tensor.shape[-1]
    # Reshape tensor to (B, H*W) and find minimum value along axis 1
    ###Change .view parametes if want to allow for multichannel images###
    minVals, _ = tensor.view(-1, N*N).min(axis=1) 
    minTensor = torch.ones_like(tensor)
    # Iterate over batches
    for i in range(minVals.shape[0]):
        # Assing minimum pixel value to each element of the image
        minTensor[i] = minTensor[i]*minVals[i]
    return minTensor

def _findMax(tensor):
    """
    Find maximum value of each batch of image tensor of shape (B, ..., H, W) and
    return tensor of same shape containing maximum value at each element for each
    batch

    Parameters
    ----------
    tensor: torch.FloatTensor
       Input tensor  

    Returns
    -------
    maxTensor: torch.FloatTensor
        Tensor containing the maximum value of an image at each batch.
        Returns tensor of same shape as input

    Note
    ----
    Function assumes square images, hence H == W
    Function assumes only one colour channel
    """
    # Size of image
    N = tensor.shape[-1]
    # Reshape tensor to (B, H*W) and find minimum value along axis 1
    ###Change .view parametes if want to allow for multichannel images###
    maxVals, _ = tensor.view(-1, N*N).max(axis=1)
    maxTensor = torch.ones_like(tensor)
    # Iterate over batches
    for i in range(maxVals.shape[0]):
        # Assing minimum pixel value to each element of the image
        maxTensor[i] = maxTensor[i]*maxVals[i]
    return maxTensor

def normaliseTensor(input):
    """
    Normalise image tensor that contains batches. Shape of input is (B, ..., H, W)

    Parameters
    ----------
    input: torch.FloatTensor
        Image tensor to be normalised
    
    Returns
    -------
    output: torch.FloatTensor
        Normalised image tensor

    Note
    ----
    Function assumes square images, hence H == W
    Function assumes only one colour channel
    """
    return (input-_findMin(input))/(_findMax(input) - _findMin(input))

def adjustArray(array):
    """
    Normalise ndArray 

    Parameters
    ----------
    input: ndArray
        array to be normalised
    
    Returns
    -------
    output: ndArray
        Normalised array
    """
    return (array) / (array.max() - array.min())

def gradientPenalty(discriminator, realImage, fakeImage, device=torch.device("cpu")):
    """
    Compute gradient penalty to be appled to Wasserstein-GAN's adverserial loss
    in order to apply Lipchitz constraint

    Parameters
    ----------
    discriminator: torch.nn.Module
        discriminator neural network of a WGAN

    realImage: torch.FloatTensor
        Tensor constaining ground truth image

    fakeImage: torch.FloatTensor
        Tensor containing an image created by a generator neural network

    device: torch.Device, optional
        Device where genereted tensors should be strored

    Returns
    -------
    gradientPenalty: torch.FloatTensor
        Gradient penalty to be applied to the adverserial loss
    """
    B, C, H, W = realImage.shape  # Batch, Channel, Hight, Width

    # Create interpolated images (mix of real and fake with some random weight)
    epsilon = torch.rand((B, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolatedImages = realImage * epsilon + fakeImage * (1 - epsilon)

    # Calculate discriminator score
    mixedScores = discriminator(interpolatedImages)

    gradient = torch.autograd.grad(inputs=interpolatedImages,
                                   outputs=mixedScores,
                                   grad_outputs=torch.ones_like(mixedScores),
                                   create_graph=True,
                                   retain_graph=True)[0]

    # Calculate gradient penalty from gradients discriminator value of interpolated 
    # image
    gradient = gradient.view(gradient.shape[0], -1)
    gradientNorm = gradient.norm(2, dim=1)
    gradientPenalty = torch.mean((gradientNorm-1)**2)
    
    return gradientPenalty
