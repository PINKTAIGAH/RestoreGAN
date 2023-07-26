import numpy as np
import torch
from ImageGenerator import ImageGenerator
import matplotlib.pyplot as plt

class JitterFilter(object):
    
    def __init__(self):
        pass

    def rowJitter(self, array, hight, jitterRadius):
        arrayCopy = np.copy(array)
        jitterRadiusList = np.arange(-jitterRadius, jitterRadius+1)

        self.jitterVector = np.random.choice(jitterRadiusList, size=hight)

        for idx in range(hight):
            arrayCopy[idx] = np.roll(arrayCopy[idx], self.jitterVector[idx])
        return arrayCopy

    def printJitterVector(self):
        print(self.jitterVector)

    def rowDejitterBatch(self, batch, vector):
        batchCopy = torch.clone(batch)
        batchSize = batchCopy.shape                        # N * C * M * M

        for i in range(batchSize[0]):                  # N
            for j in range(batchSize[1]):              # C
                for k in range(batchSize[-1]):         # M
                    batchCopy[i][j][k] = torch.roll(batchCopy[i][j][k], 
                                                    vector[i][k].item()) 
        return batchCopy

if __name__ == "__main__":
    Filter = JitterFilter()
    Generator = ImageGenerator(N=128)
    
    k5 = Generator.genericNoise(kernalSize=5)
    k5Jittered = Filter.rowJitter(k5, 128, 5) 

    k65 = Generator.genericNoise(kernalSize=65)
    k65Jittered = Filter.rowJitter(k65, 128, 5) 

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(k5, cmap="gray")
    ax2.imshow(k5Jittered, cmap="gray")
    ax3.imshow(k65, cmap="gray")
    ax4.imshow(k65Jittered, cmap="gray")
    plt.show()
