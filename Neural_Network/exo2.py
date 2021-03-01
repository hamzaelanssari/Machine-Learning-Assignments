# program1
import matplotlib.pyplot as plt
from random import seed
from random import random
import numpy as np


def f(x):
    return -x + 5


def generateData(n):
    # generate random numbers between 0-1
    seed(1)
    min = 0
    max = 5
    dataset = []
    for _ in range(n):
        valueX1 = random()
        scaledvalueX1 = min + (valueX1 * (max - min))
        valueX2 = random()
        scaledvalueX2 = min + (valueX2 * (max - min))
        x1 = round(scaledvalueX1, 1)
        x2 = round(scaledvalueX2, 1)
        y0 = f(x1)
        cl = 0
        if (x2 > y0):
            cl = 1
        dataset.append([x1, x2, cl])
    return dataset


def plotData(dataset):
    c0 = dataset[:, 2] == 0
    c1 = dataset[:, 2] == 1
    XD1 = dataset[:, 0][c0]
    YD1 = dataset[:, 1][c0]
    XD2 = dataset[:, 0][c1]
    YD2 = dataset[:, 1][c1]
    plt.plot(XD1, YD1, 'rD')
    plt.plot(XD2, YD2, 'bo')
    plt.show()


dataset = np.array(generateData(10))
data=dataset[:,2]
print(data)
plotData(dataset)