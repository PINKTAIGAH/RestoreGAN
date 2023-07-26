import torch
import torch.nn as nn
from torch.nn.modules import padding

"""
Model for Discriminator
"""

class Discriminator(nn.Module):

    def __init__(self, channelImages, featuresD):
        super(Discriminator, self).__init__()
        self.critic = nn.Sequential(
            ### INPUT SIZE: N * channelImages * 128 * 128 
            nn.Conv2d(channelImages, featuresD, kernel_size=4,
                      stride=2, padding= 1),
            ### SIZE: 64*64
            nn.LeakyReLU(0.2),
            self._block(featuresD, featuresD*2, 4, 2, 1),
            ### SIZE: 32*32
            self._block(featuresD*2, featuresD*4, 4, 2, 1),
            ### SIZE: 16*16
            self._block(featuresD*4, featuresD*8, 4, 2, 1),
            ### SIZE: 8*8
            self._block(featuresD*8, featuresD*16, 4, 2, 1),
            ### SIZE: 4*4
            nn.Conv2d(featuresD*16, 1, kernel_size=4, stride=2, padding=0),
            ### SIZE: 1*1
        )
    
    ### Convolutional block that contains BatchNorm2d after a convolution
    def _block(self, inChannels, outChannels, kernalSize, stride, padding):
        ### This will be internal blocks of layers of the model
        ### In and out channels are the channel input and output size
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernalSize, 
                      stride, padding, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.critic(x)

"""
Initialise weights for discriminator
"""

def initialiseWeights(model):
    ### Setting init weights when doing one of the following operations
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N = 128
    x = torch.randn((5, 1, N, N))
    disc = Discriminator(1, 32)
    initialiseWeights(disc)
    y = disc(x)
    print(y.shape)


if __name__ == "__main__":
    test()
