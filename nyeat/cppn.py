import networkx as nx
import numpy as np


class CPPN(object):

    def __init__(self, nn, use_bias=True, use_radius=True):
        self.nn = nn
        self.use_bias = use_bias
        self.use_radius = use_radius

    def render(self, grid_shape, ranges=None):
        """
        Applies the model to a linear grid of coordinates
        and radial distances at a given resolution.
        """
        if ranges is None:
            ranges = [(-1., 1.) for _ in range(len(grid_shape))]
        # rectangular coordinates
        inputs = np.meshgrid(
            *[
                np.linspace(low, high, n)
                for n, (low, high) in zip(grid_shape, ranges)
            ], indexing='ij'
        )
        num_missing_inputs = len(self.nn.input_nodes) - len(inputs) - self.use_bias - self.use_radius
        pad_inputs = [np.zeros(grid_shape) for _ in range(num_missing_inputs)]
        #inputs = pad_inputs + inputs
        inputs = inputs + pad_inputs
        indexes_arr = np.array(inputs)
        # radius
        if self.use_radius:
            index_r = np.sqrt(np.sum(np.square(indexes_arr), axis=0))
            inputs.append(index_r)
        # bias
        if self.use_bias:
            inputs.append(np.ones(indexes_arr.shape[1:]))
        #inputs = np.array(inputs)
        #print('cppn1:', indexes_arr.shape, inputs.shape)
        y = self.nn.eval(inputs)
        return y
