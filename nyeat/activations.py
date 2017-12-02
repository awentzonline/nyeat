import numpy as np


def gaussian(x):
    return np.exp(0.5 * x ** 2) / np.sqrt(2 * np.pi)


def relu(x):
    return np.clip(x, 0., None)


def softmax(x, t=1.):
    e_x = np.exp((x - np.max(x)) / t)
    return e_x / e_x.sum()


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))
