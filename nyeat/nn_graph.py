import networkx as nx
import numpy as np


class NNGraph(object):
    def __init__(self, graph, input_nodes, output_nodes):
        self.g = graph
        self._sorted_nodes = list(nx.topological_sort(self.g))
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

    def eval(self, inputs, dtype=np.float32):
        activations = {}
        # assign inputs
        for i, v in enumerate(inputs):
            activations[i] = v
        # activate network
        g_edges = self.g.edges
        g_in_edges = self.g.in_edges
        for node in self._sorted_nodes:
            if node in activations:
                continue
            in_edges = g_in_edges(node)
            activation = np.sum(
                activations[in_node] * g_edges[in_node, out_node]['g'].weight
                for in_node, out_node in in_edges
            )
            node_dict = self.g.nodes[node]
            if node_dict['n'].f_activation is not None:
                activation = node_dict['n'].f_activation(activation)
            activations[node] = activation
        return np.array(
            [activations[node] for node in self.output_nodes],
            dtype=dtype)
