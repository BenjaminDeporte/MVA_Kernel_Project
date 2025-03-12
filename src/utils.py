import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def recast_y(y):
    return 2*y-1
