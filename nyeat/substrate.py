import numpy as np


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
