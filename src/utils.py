import torch
import config
from torchvision.utils import save_image
import torch.nn.functional as F

"""
Input/Output
"""

def save_some_examples(gen, val_loader, epoch, folder, filter):
    x, y , vector= next(iter(val_loader))
    x, y, vector = x.to(config.DEVICE), y.to(config.DEVICE), vector.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        vector_fake = gen(x)
        y_fake = filter.rowDejitterBatch(x, vector_fake)

        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

"""
Input manipulation
"""

def tensorConcatinate(tensorLeft, tensorRight):
    tensorRight = tensorRight.view(-1, tensorRight.shape[-1])
    tensorLeft = tensorLeft.view(-1, tensorLeft.shape[-1])
    return torch.cat((tensorLeft, tensorRight), dim=1) 

def verticalEdges(inputTensor):
    print(inputTensor.shape)
    print(config.SOBEL_KERNAL.shape)
    return F.conv2d(inputTensor.reshape(1, 1, *inputTensor.shape),
                    config.SOBEL_KERNAL.reshape(1, 1, *config.SOBEL_KERNAL.shape),
                    padding="same")

def findMin(tensor):
    N = tensor.shape[-1]
    minVals, _ = tensor.view(-1, N*N).min(axis=1)
    minTensor = torch.ones_like(tensor)
    for i in range(minVals.shape[0]):
       minTensor[i] = minTensor[i]*minVals[i]
    return minTensor


def findMax(tensor):
    N = tensor.shape[-1]
    maxVals, _ = tensor.view(-1, N*N).max(axis=1)
    maxTensor = torch.ones_like(tensor)
    for i in range(maxVals.shape[0]):
       maxTensor[i] = maxTensor[i]*maxVals[i]
    return maxTensor


def rescaleTensor(tensor):
    return (tensor-findMin(tensor))/findMax(tensor)


"""
WGAN graident penalty
"""

def gradientPenalty(discriminator, realImage, fakeImage, device=torch.device("cpu") ):

    BATCH_SIZE, C, H, W = realImage.shape  # Channel, Hight, Width

    ### Create interpolated images (mix of real and fake with some random weight)
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolatedImages = realImage * epsilon + fakeImage * (1 - epsilon)

    ### Calculate critic score
    mixedScores = discriminator(interpolatedImages)

    gradient = torch.autograd.grad(inputs=interpolatedImages,
                                   outputs=mixedScores,
                                   grad_outputs=torch.ones_like(mixedScores),
                                   create_graph=True,
                                   retain_graph=True)[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradientNorm = gradient.norm(2, dim=1)
    gradientPenalty = torch.mean((gradientNorm-1)**2)
    
    return gradientPenalty
