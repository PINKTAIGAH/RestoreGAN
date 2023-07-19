import torch

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

