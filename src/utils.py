import torch
import config
from torchvision.utils import save_image

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

"""
Model saving/loading
"""

def saveCheckpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def loadCheckpoint(checkpointFile, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpointFile, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for paramGroup in optimizer.paramGroups:
        paramGroup["lr"] = lr


"""
Saving examples of generated and ground truth images for easy viewing
"""


def saveSomeExamples(generator, validationLoader, epoch, folder):
    x, y = next(iter(validationLoader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    generator.eval()
    with torch.no_grad():
        yFake = generator(x)
        yFake = yFake * 0.5 + 0.5  # remove normalization#
        save_image(yFake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    generator.train()
