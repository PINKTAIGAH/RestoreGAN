import numpy as np

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

if __name__ == "__main__":
    Filter = JitterFilter()
    print(Filter.rowJitter(np.arange(100).reshape(10, 10), 10, 3))
    Filter.printJitterVector()
