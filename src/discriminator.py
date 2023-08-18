import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    A torch.nn.Module instance containing a Deep Convolutional GAN network designed
    to discriminate images of size 128*128. Architecture of this network is based
    on the folliwing paper (https://doi.org/10.48550/arXiv.1511.06434)

    Atribute
    --------
    critic: torch.nn.Sequential instance
        Object that will return the output of the DCGAN discriminator

    Parameters
    ----------
    channelImages: int
        Number of colour channels in discriminator input

    featuresD: int
        Coefficient that will scale the number of channels created at each 
        convolutional block of the discriminator
    """

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
        """
        Convolutional block that contains a batch normalisation after the 
        convolution of the input tensor.

        Parameters
        ----------
        inChannels: int
            Number of image channels in input image

        outChannels: int
            Number of image channels that output image will have

        kernalSize: int
            Hight and width of convolutional kernal

        stride: int
            Stride of convolution

        padding: int
            Hight and width of padding to be added to image prior to convolution

        Returns
        -------
        output: torch.nn.Sqeuential instance
            Instance of Sequential class containing all steps required for 
            convolutional block

        Notes
        -----
        These convolutional blocks contain a batch normalisation. This is inline
        with the internal layers of the DCGAN model, as oposed to the initial and 
        final convilutional blocks (which do not contain a batch normalisation)
        """
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernalSize, 
                      stride, padding, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        """
        Returns output of DCGAN discriminator model when called

        Parameters
        ----------
        x: torch.FloatTensor
            Input tensor to be passed through discriminator

        Returns
        -------
        output: torch.FloatTensor
            Tensor containing discriminator score of the inputted image
        """
        return self.critic(x)

def initialiseWeights(model):
    """
    Initialise weights of discriminator  

    Parameters
    ----------
    model: torch.nn.Module instance
        Model object whose weights will be initialised 
    """
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
