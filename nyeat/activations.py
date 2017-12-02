import numpy as np


def gaussian(x):
    return np.exp(0.5 * x ** 2) / np.sqrt(2 * np.pi)


def relu(x):
    return np.clip(x, 0., None)
