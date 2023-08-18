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
                      stride=2, padding=0),
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
            nn.Tanh(),          # CHANGE TO EXPANDED TANH
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

class FullyConnected(nn.Module):
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

    def __init__(self,inChannel, inFeatures, outFeatures=256):
        super().__init__()

        self.fullyConnected = nn.Sequential(
            nn.Flatten(start_dim=1),
            
            nn.Linear(in_features=inChannel*inFeatures,
                      out_features=2*outFeatures),
            nn.ReLU(),
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

    down1: torch.nn.Sequential instance
        Object which returns the output of the first two convolutional blocks of 
        RestoreGAN's generator

    down2: torch.nn.Sequential instance
        Object which returns the output of the third convolutional blocks of 
        RestoreGAN's generator

    down3 & down4: torch.nn.Sequential instance
        Object which returns the output of the Resnet convolutional blocks of 
        RestoreGAN's generator

    down5: torch.nn.Sequential instance
        Object which returns the output of the sixth convolutional block of 
        RestoreGAN's generator & Tanh activation function

    down6: torch.nn.Sequential instance
        Object which returns the output of the fully connected layer of the 
        RestoreGAN generator

    Parameters
    ----------
    imageSize: int
        Hight and width of input image tensor

    inChannel: int
        Number of colour channels of input image

    outChannels: int
        Number of output channels of sixth convolutional block. Corresponds to 
        number of elements in output flow map vectors

    Notes
    -----
    Architecture of this network is based on the following paper 
    (https://doi.org/10.3390/s21144693)
    """
    def __init__(self, imageSize, inChannel=1, outChannel=2,):
        super().__init__()

        self.fullyConnectedFeatures = int(imageSize/4 - 10)
        self.imageSize = imageSize
        self.outChannel = outChannel
        
        ### Input channels ==> 1
        self.down1 = LargeBlock(inChannel=inChannel)
        ### out channels ==> 64  
        self.down2 = ThirdBlock(inChannel=64) 
        ### out channels ==> 128
        self.down3 = ResBlock(inChannel=128)
        ### out channel ==> 128 + 128 from  skip connection
        self.down4 = ResBlock(inChannel=128)
        ### out channel ==> 128 + 128 from skip connection
        self.down5 = SixthBlock(inChannel=128, outChannel=outChannel,) 
        ### out channels = 2
        self.down6 = FullyConnected(inChannel=outChannel,
                                    inFeatures=self.fullyConnectedFeatures**2,
                                    outFeatures=self.imageSize**2,)
        ### out shape = B * N * N * 2 (Shape of flow map for unshifting) 
    
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
        d1 = self.down1(x)
        # print(f"d1 shape ==> {d1.shape}")
        d2 = self.down2(d1)
        # print(f"d2 shape ==> {d2.shape}")
        d3 = self.down3(d2)
        # print(f"d3 shape ==> {d3.shape}")
        d4 = self.down4(d3)
        # print(f"d4 shape ==> {d4.shape}")
        d5 = self.down5(d4)
        # print(f"d5 shape ==> {d5.shape}")
        d6 = self.down6(d5)

        output = torch.reshape(
            d6, (d6.shape[0], self.imageSize, self.imageSize, self.outChannel)
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
    x = torch.randn((16, 1, N, N))
    ideal = torch.rand((16, N, N, 2))
    model = Generator(imageSize=N, inChannel=1,
                      outChannel=2)
    initialiseWeights(model)
    predicition = model(x)
    print(predicition.shape)
    print(ideal.shape)

if __name__ == "__main__":
    test()
