import numpy as np
from scipy.fftpack import idct


class Substrate(object):
    def __init__(self, input_dims, layer_confs, dtype=np.float32):
        self.dtype = dtype
        self.matrices = []
        self.activations = []
        last_ndims = input_dims
        for ndims, activation in layer_confs:
            shape = (ndims, last_ndims + 1)
            self.matrices.append(
                np.zeros(shape).astype(dtype))
            self.activations.append(activation)
            last_ndims = ndims

    def update_weights(self, cppn):
        for layer_i, matrix in enumerate(self.matrices):
            # TODO: do this with only one call to render?
            values = cppn.render(matrix.shape)
            values = values[layer_i].T
            matrix[:] = values

    def eval(self, x):
        for matrix, activation in zip(self.matrices, self.activations):
            matrix, bias = matrix[:, :-1], matrix[:, -1]
            x = np.dot(matrix, x) + bias
            if activation:
                x = activation(x)
        return x

    @property
    def num_layers(self):
        return len(self.matrices)


class DCTSubstrate(Substrate):
    def update_weights(self, cppn):
        for layer_i, matrix in enumerate(self.matrices):
            # try to start with high values in the upper-left corner
            ranges = ((2 * np.pi, 0.),) * len(matrix.shape)
            values = cppn.render(matrix.shape, ranges=ranges)
            values = idct(values[layer_i].T)
            matrix[:] = values


class NeuralNetwork1DSubstrate(object):
    """A multi-spatial substrate that represents fully-connected
    neural network.

    The each dimension of the CPPN output is for a different
    weight matrix of a fully-connected neural network.
    """
    def __init__(self, input_dims, layer_confs, dtype=np.float32):
        self.dtype = dtype
        self.matrices = []
        self.biases = []
        self.activations = []
        last_ndims = input_dims
        for ndims, activation in layer_confs:
            shape = (ndims, last_ndims)
            self.matrices.append(
                np.zeros(shape).astype(dtype))
            self.biases.append(
                np.zeros((ndims,)).astype(dtype))
            self.activations.append(activation)
            last_ndims = ndims

    def update_weights(self, cppn):
        num_layers = len(self.matrices)
        for layer_i, matrix in enumerate(self.matrices):
            # TODO: do this with fewer calls to render?
            # Multiple calls are because the matrix shapes vary
            values = cppn.render(matrix.shape)
            values = values[layer_i * 2]
            matrix[:] = values
            bias = self.biases[layer_i]
            values = cppn.render(bias.shape)
            values = values[layer_i * 2 + 1]
            bias[:] = values

    def eval(self, x):
        x = x.ravel()
        for matrix, bias, activation in zip(self.matrices, self.biases, self.activations):
            #matrix, bias = matrix[:, :-1], matrix[:, -1]
            x = np.dot(matrix, x) + bias
            if activation:
                x = activation(x)
        return x


class NeuralNetwork2DSubstrate(object):
    """A multi-spatial substrate that represents fully-connected
    neural network.

    The each dimension of the CPPN output is for a different
    weight matrix of a fully-connected neural network.
    """
    def __init__(self, input_shape, layer_confs, dtype=np.float32):
        self.dtype = dtype
        self.matrices = []
        self.biases = []
        self.activations = []
        last_shape = input_shape
        for this_shape, activation in layer_confs:
            shape = this_shape + last_shape
            self.matrices.append(
                np.zeros(shape).astype(dtype))
            self.biases.append(
                np.zeros(this_shape).astype(dtype))
            self.activations.append(activation)
            last_shape = this_shape

    def update_weights(self, cppn):
        num_layers = len(self.matrices)
        for layer_i, matrix in enumerate(self.matrices):
            # TODO: do this with fewer calls to render?
            # Multiple calls are because the matrix shapes vary
            shape = matrix.shape
            while len(shape) < len(matrix.shape):
                shape = (1,) + shape
            values = cppn.render(shape)
            while len(shape) != len(matrix.shape):
                values = values[0, :]
            values = values[layer_i * 2]
            #print('upw', layer_i, values.shape, matrix.shape)
            matrix[:] = values
            bias = self.biases[layer_i]
            values = cppn.render(bias.shape)
            values = values[layer_i * 2 + 1]
            bias[:] = values
            #print('upw:', matrix.min(), matrix.mean(), matrix.max(), matrix.shape)

    def eval(self, x):
        x = x.ravel()
        for matrix, bias, activation in zip(self.matrices, self.biases, self.activations):
            #matrix, bias = matrix[:, :-1], matrix[:, -1]
            shape = matrix.shape
            if len(shape) > 2:
                #shape = (np.prod(shape[:-1]), shape[-1])
                shape = (shape[0], np.prod(shape[1:]))
                matrix = matrix.reshape(shape)
            #print('ssev:', x.shape, matrix.shape, bias.shape)
            x = np.dot(matrix, x) + bias
            if activation:
                x = activation(x)
            #print('ssev_result:', x.shape)
        return x
