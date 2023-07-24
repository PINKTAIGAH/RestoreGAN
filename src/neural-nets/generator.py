import torch 
import torch.nn as nn

class LargeBlock(nn.Module):
    
    def __init__(self, inChannel=1):
        super().__init__()
        
        self.firstBlock = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=64, kernel_size= 11, 
                      stride=1, padding=0,), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.secondBlock = nn.Sequential(
            nn.MaxPool2d(kernel_size=7, stride=1, padding=0,),
            nn.Conv2d(in_channels= 64, out_channels=64, kernel_size=7, stride=1, 
                      padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.firstBlock(x)
        return self.secondBlock(x)
        

class ThirdBlock(nn.Module):

    def __init__(self, inChannel=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=128, kernel_size=3,
                      stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
     
    def __init__(self, inChannel=128):
        super().__init__()

        self.resBlock = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=128, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(128),
        )

    def forward(self, x):
        return self.resBlock(x)

class SixthBlock(nn.Module):

    def __init__(self, inChannel=256, A=1):
        super().__init__()

        self.sixthBlock = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=2, kernel_size=5, stride=1,
                      padding=0),
            nn.BatchNorm2d(2),
            nn.Tanh(),          #CHANGE TO EXPANDED TANH
        )

        self.A = A

    def forward(self, x):
        return self.sixthBlock(x)*self.A

class Generator(nn.Module):
    def __init__(self, inChannel=1):
        super().__init__()
        
        ### Input channels ==> 1
        self.down1 = LargeBlock(inChannel=inChannel)
        ### out channels ==> 64  
        self.down2 = ThirdBlock(inChannel=64) 
        ### out channels ==> 128
        self.down3 = ResBlock(inChannel=128)
        ### out channel ==> 128 + 128 from  skip connection
        self.down4 = ResBlock(inChannel=128*2)
        ### out channel ==> 128 + 128 from skip connection
        self.down5 = SixthBlock(inChannel=128*2)
        ### out channels = 2
    

    def forward(self, x):

        d1 = self.down1(x)
        print(f"d1 shape ==> {d1.shape}")

        d2 = self.down2(d1)
        print(f"d2 shape ==> {d2.shape}")

        d3 = self.down3(d2)
        print(f"d3 shape ==> {d3.shape}")

        d4 = self.down4(torch.cat([d3, d2], 1))
        print(f"d4 shape ==> {d4.shape}")

        d5 = self.down5(torch.cat([d4, d3], 1))
        print(f"d5 shape ==> {d5.shape}")

        return d5 

def test():
    x = torch.randn((1, 1, 256, 256))
    model = Generator(inChannel=1)
    predicition = model(x)
    print(predicition.shape)

if __name__ == "__main__":
    test()
