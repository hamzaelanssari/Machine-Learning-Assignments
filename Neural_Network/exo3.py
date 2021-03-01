import math
import numpy as np

def sigmoid(z):
    return 1. / (1 + math.exp(-z))


def predict(x1, x2, w1, w2, bias):
    z = bias + x1 * w1 + x2 * w2
    return [0 if sigmoid(z)<0.5 else 1]


def predictV2(X, W, bias):
    Z = bias + np.dot(X, W)
    return [0 if sigmoid(z)<0.5 else 1 for z in Z]


print(predict(0.7, 4.2, 0.1, 0.2, 0.3))

XTest = [[0.7, 4.2], [3.8, 1.3], [2.5, 2.2], [3.3, 3.9], [0.5, 0.1]]
bias = -1.22
W = [0.35, 0.19]
print(predictV2(XTest, W, bias))