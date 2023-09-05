import torch 
import torch.nn as nn

class LargeBlock(nn.Module):
    """
    An instance of the torch.nn.Module class containing the first two convolutional 
    blocks of the RestoreGAN's generator neural network

    Atributes
    ---------
    firstBlock: torch.nn.Sequential instance
        Object that will return the output of the first large kernal convolutional
        block of the RestoreGAN network (kernal size 11)

    secondBlock: torch.nn.Sequential instance
        Object that will return the output of the second large kernal convolutional
        block of the RestoreGAN network (kernal size 7)

    Parameters
    ----------
    inChannel: int, optional
        Number of colour channels of input image 

    outChannel: int, optional
        Number of image chanels of output tensor of the second convolutional block

    middleChannel: int, optional
        Number of image chanels of output tensor of the first convolutional block
    """
    
    def __init__(self, inChannel=1, outChannel=64, middleChannel=64):
        super().__init__()
        
        self.firstBlock = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=middleChannel,
                      kernel_size= 11, stride=1, padding=0,), 
            nn.BatchNorm2d(middleChannel),
            nn.ReLU(),
        )

        self.secondBlock = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,),
            nn.Conv2d(in_channels= middleChannel, out_channels=outChannel, 
                      kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Returns output of first two large kernal convolutional locks of the 
        RestoreGAN's generator network when called

        Parameters
        ----------
        x: torch.FloatTensor
            Input tensor to be passed through convolutional block

        Returns
        -------
        output: torch.FloatTensor
            Tensor containing output of first two convolutional blocks
        """
        x = self.firstBlock(x)
        return self.secondBlock(x)
        

class ThirdBlock(nn.Module):
    """
    An instance of the torch.nn.Module class constaining the third convolutional 
    blocks of the RestoreGAN's generator neural network

    Atributes
    ---------
    block: torch.nn.Sequential instance
        Object that will return the output of the third convolutional block of 
        the RestoreGAN network 

    Parameters
    ----------
    inChannel: int, optional
        Number of image channels of input tensor 

    outChannel: int, optional
        Number of image chanels of output tensor
    """

    def __init__(self, inChannel=64, outChannel=128):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Returns output of third convolutional bocks of the RestoreGAN's generator 
        network when called

        Parameters
        ----------
        x: torch.FloatTensor
            Input tensor to be passed through convolutional block

        Returns
        -------
        output: torch.FloatTensor
            Tensor containing output of third convolutional blocks
        """
        return self.block(x)

class ResBlock(nn.Module):
    """
    An instance of the torch.nn.Module class containing a ResNet residual  
    convolutional block. Output tensor will have the same shape as the input tensor 

    Atributes
    ---------
    resBlock: torch.nn.Sequential
        Object that will return the output of a residual convolutional block

    Parameters
    ----------
    inChannel: int, optional
        Number of image channels of input tensor 

    outChannel: int, optional
        Number of image chanels of output tensor

    Notes
    -----
    Architecture of Resnet block is described in (https://doi.org/10.48550/arXiv.1512.03385)
    """ 
    def __init__(self, inChannel=128, outChannel=128):
        super().__init__()

        self.resBlock = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=128, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=outChannel, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(outChannel),
        )

    def forward(self, x):
        """
        Returns output of single resNet resudual block when called

        Parameters
        ----------
        x: torch.FloatTensor
            Input tensor to be passed through convolutional block

        Returns
        -------
        output: torch.FloatTensor
            Tensor containing output of Resnet convolutional block 
        """
        # Identity refers to input tensor
        identity = x
        x = self.resBlock(x)
        # Adding output of resBlock to identity to calculate residual
        x+= identity
        return x

class SixthBlock(nn.Module):
    """
    An instance of the torch.nn.Module class containing sixth convolutional 
    blocks of the RestoreGAN's generator neural network. Output of convolution
    is passed through a Tanh activation function to restrain output to to range [-1, 1]

    Atributes
    ---------
    block: torch.nn.Sequential instance
        Object that will return the output of the third convolutional block of 
        the RestoreGAN network 

    Parameters
    ----------
    inChannel: int, optional
        Number of image channels of input tensor 

    outChannel: int, optional
        Number of image chanels of output tensor
    """
    def __init__(self, inChannel=128, outChannel=2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=5,
                      stride=1,padding=0),
            nn.BatchNorm2d(outChannel),
            nn.Tanh(),          
        )

    def forward(self, x):
        """
        Returns output of sixth convolutional block of RestoreGAN's generator 
        when called

        Parameters
        ----------
        x: torch.FloatTensor
            Input tensor to be passed through convolutional block

        Returns
        -------
        output: torch.FloatTensor
            Tensor containing output of sixth convolutional blocks
        """
        return self.block(x)

class _FullyConnected(nn.Module):
    """
    An instance of the torch.nn.Module class containing the fully connected blocks
    of the RestoreGAN's generator neural network. Output of fully connected layer
    will have a shape of (B, H, W, 2), which is the shape of a flow map as required
    by torch.nn.functional.grid_sample.

    Atributes
    ---------
    block: torch.nn.Sequential instance
        Object that will return the output of the fully connected kayer of the 
        RestoreGAN network 

    Parameters
    ----------
    inChannel: int
        Number of image channels of input tensor 

    inFeatures: int
        Number of features from input tensor after flattened

    outFeatures: int, optional
        Number of output features from fully conneted layer
    """

    def __init__(self, inChannel, inFeatures, outFeatures):
        super().__init__()

        self.fullyConnected = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=inChannel*inFeatures,
                      out_features=outFeatures),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Returns output of fully connected layer of RestoreGAN's generator when called

        Parameters
        ----------
        x: torch.FloatTensor
            Input tensor to be passed through fully connected layer 

        Returns
        -------
        output: torch.FloatTensor
            Tensor containing flowmap output of RestoreGAN's generator 
        """
        return self.fullyConnected(x)

class _ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, down=True, useAct=True, **kwargs):
        super().__init__()

        # Define generic convolutional and transpose convolutional block
        self.conv = nn.Sequential(
            nn.Conv2d(
                inChannels,
                outChannels,
                padding_mode="reflect",
                **kwargs,
            )
            if down else nn.ConvTranspose2d(
                inChannels,
                outChannels,
                **kwargs,
            ),
            nn.BatchNorm2d(outChannels),
            # Only pass through activation function if useAct is True
            nn.ReLU(inplace=True) if useAct else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)

class _ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        
        # Define convolutional block of the residual blocks
        self.block = nn.Sequential(
            _ConvBlock(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            _ConvBlock(
                channels,
                channels,
                useAct=False,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

    def forward(self, x):
        # Return residual component of block
        return x + self.block(x)


class Generator(nn.Module):
    """
    A torch.nn.Module instance containing the generator of the RestoreGAN neural
    network desined to generate flow maps to unshift an input image of size 
    128*128. 

    Atributes
    ---------
    fullyConnectedFeatures: int
        Size of image form output tensor of down5 object. Calculated using a combination
        of equation 1 (see notes) for each convolutional block the tensor pass 
        passed through

    down1: torch.nn.ModuleList instance
        Object which returns the output of the first three convolutional blocks of 
        RestoreGAN's generator

    resBlock: torch.nn.Sequential instance
        Object which returns the output of the two residual blocks of 
        RestoreGAN's generator

    down3: torch.nn.ModuleList instance
        Object which returns the output of final convolutional block + fully connected
        layer of RestoreGAN's generator

    Parameters
    ----------
    imageSize: int
        Hight and width of input image tensor

    inChannel: int
        Number of colour channels of input image

    outChannels: int
        Number of output channels of sixth convolutional block. Corresponds to 
        number of elements in output flow map vectors

    numFeatures: int
        Coefficient used to compute the input and output channels of the hidden 
        layers of the generator

    numResiduals: int
        Number of residual blocks used in the generator

    fullyConnectedFeatures: int
        Width of input for fully connected layer

    Notes
    -----
    Architecture of this network is based on the following paper 
    (https://doi.org/10.3390/s21144693)
    """
    def __init__(self, imageSize, inChannels=1, outChannels=1, numFeatures=64,
                 numResiduals=2, fullyConnectedFeatures=60):
        super().__init__()

        self.fullyConnectedFeatures = fullyConnectedFeatures
        self.imageSize = imageSize
        self.outChannels = outChannels
        
        self.down1 = nn.ModuleList([
            _ConvBlock(inChannels, numFeatures, kernel_size=11, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            _ConvBlock(numFeatures, numFeatures, kernel_size=7, stride=1, padding=3),
            _ConvBlock(numFeatures, numFeatures*2, kernel_size=3, stride=1, padding=1),
        ])

        self.resBlock= nn.Sequential(
            *[_ResidualBlock(numFeatures*2) for _ in range(numResiduals)] 
        )

        self.down2 = nn.ModuleList([
            _ConvBlock(numFeatures*2, outChannels, kernel_size=5, stride=1, padding=1),
            nn.Tanh(),
            _FullyConnected(outChannels, self.fullyConnectedFeatures**2, 2*self.imageSize**2)
        ])

    def forward(self, x):
        """
        Returns output of RestoreGAN's generator model when called

        Parameters
        ----------
        x: torch.FloatTensor
            Input tensor to be passed through discriminator

        Returns
        -------
        output: torch.FloatTensor
            Tensor containing discriminator score of the inputted image
        """
        for layer in self.down1:
            x = layer(x)

        x = self.resBlock(x)

        for layer in self.down2:
            x = layer(x)
        
        output = torch.reshape(
            x, (x.shape[0], self.imageSize, self.imageSize, self.outChannels)
        )

        return output

def initialiseWeights(model):
    """
    Initialise weights of generator  

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
    x = torch.randn((1, 1, N, N))
    ideal = torch.rand((1, N, N, 1))
    model = Generator(imageSize=N, inChannels=1,
                      outChannels=2)
    initialiseWeights(model)
    predicition = model(x)
    print(predicition.shape)
    print(ideal.shape)

if __name__ == "__main__":
    test()
