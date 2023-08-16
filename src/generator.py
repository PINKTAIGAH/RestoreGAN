import config
import torch 
import torch.nn as nn

class LargeBlock(nn.Module):
    
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
        x = self.firstBlock(x)
        return self.secondBlock(x)
        

class ThirdBlock(nn.Module):

    def __init__(self, inChannel=64, outChannel=128):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=3,
                      stride=2, padding=0),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
     
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
        identity = x
        x = self.resBlock(x)
        x+= identity
        return x

class SixthBlock(nn.Module):

    def __init__(self, inChannel=128, outChannel=2):
        super().__init__()

        self.sixthBlock = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=5,
                      stride=1,padding=0),
            nn.BatchNorm2d(outChannel),
            nn.Tanh(),          # CHANGE TO EXPANDED TANH
        )

    def forward(self, x):
        return self.sixthBlock(x)

class FullyConnected(nn.Module):

    def __init__(self,inChannel, inFeatures, outFeatures=256):
        super().__init__()

        self.fullyConnected = nn.Sequential(
            nn.Flatten(start_dim=1),
            
            nn.Linear(in_features=inChannel*inFeatures,
                      out_features=2*outFeatures),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fullyConnected(x)

class Generator(nn.Module):
    def __init__(self, imageSize, inChannel=1, outChannel=2,):
        super().__init__()

        self.inputFeatures = int(imageSize/4 - 10)
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
                                    inFeatures=self.inputFeatures**2,
                                    outFeatures=self.imageSize**2,)
        ### out shape = B * N * N * 2 (Shape of flow map for unshifting) 
    
    def forward(self, x):

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
    ### Setting init weights when doing one of the following operations
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
