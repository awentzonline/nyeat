from collections import defaultdict

import networkx as nx
import numpy as np


class NNGraph(object):
    """Traverses a DAG of function applications"""
    def __init__(self, graph, input_nodes, output_nodes):
        self.g = graph
        try:
            self._sorted_nodes = list(nx.topological_sort(self.g))
        except:
            print("BAD GRAPH?")
            print(graph)
            raise
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

    def eval(self, inputs, dtype=np.float32):
        activations = defaultdict(lambda: np.zeros(inputs[0].shape))
        # assign inputs
        input_shape = inputs[0].shape
        for i, v in enumerate(inputs):
            #print('input', i, 'val', v)
            activations[i] = v
        # activate network
        g_edges = self.g.edges
        g_in_edges = self.g.in_edges
        for node in self._sorted_nodes:
            #print('node', node)
            if node in activations:
                continue
            in_edges = g_in_edges(node)
            if in_edges:
                activation = np.sum(
                    (activations[in_node] * g_edges[in_node, out_node]['g'].weight
                    for in_node, out_node in in_edges),
                    axis=0
                )
                node_dict = self.g.nodes[node]
                if node_dict['n'].f_activation is not None:
                    activation = node_dict['n'].f_activation(activation)
            else:
                #print('no inputs for', node)
                activation = np.zeros(input_shape)
            activations[node] = activation
        try:
            output_activations = [activations[node] for node in self.output_nodes]
            return np.array(output_activations, dtype=dtype)
        except Exception as e:
            print(acts)
            raise

    @classmethod
    def from_genome(cls, genome):
        return cls(genome.to_graph(), genome.input_nodes, genome.output_nodes)
