import copy

import numpy as np

from nyeat.genome import EdgeGene, Genome, NodeGene


class NEAT(object):
    def __init__(self, edge_class=EdgeGene, node_class=NodeGene):
        self.edge_class = edge_class
        self.node_class = node_class
        self.next_innovation_number = 0
        self.innovations = []
        self.species = []
        self.genomes = []
        self.nodes = set()
        self.node_map = {}
        self.gene_map = {}

    def populate(self, population_size, num_inputs, num_outputs):
        for pop_i in range(population_size):
            genome = Genome()
            self.genomes.append(genome)
            for output_i in range(num_outputs):
                for input_i in range(num_inputs):
                    gene, in_node, out_node = self.make_gene(input_i, num_inputs + output_i)
                    out_node.f_activation = np.tanh
                    genome.add_gene(gene, in_node, out_node)

    def make_node(self, id):
        if id in self.node_map:
            return self.node_map[id]
        node = self.node_class(id)
        self.node_map[id] = node
        return node

    def make_gene(self, a, b):
        key = (a, b)
        if key in self.gene_map:
            gene = self.gene_map[key]
        else:
            gene = self.edge_class(self.innovation_number, a, b)
            self.gene_map[key] = gene
        nodes = []
        for node_id in key:
            nodes.append(self.make_node(node_id))
        return gene, nodes[0], nodes[1]

    @property
    def innovation_number(self):
        return len(self.gene_map)

    @property
    def next_node_id(self):
        if self.node_map:
            return max(self.node_map) + 1
        return 0


if __name__ == '__main__':
    neat = NEAT()
