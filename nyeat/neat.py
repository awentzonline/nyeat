import copy
from collections import defaultdict

import numpy as np

from nyeat import activations
from nyeat.genome import EdgeGene, Genome, NodeGene
from nyeat.nn_graph import NNGraph


class NEAT(object):
    activations = (
        None, activations.gaussian, activations.relu,
        np.cos, np.sin, np.abs, np.tanh,)

    def __init__(self, edge_class=EdgeGene, node_class=NodeGene, activations=None):
        self.edge_class = edge_class
        self.node_class = node_class
        self.next_innovation_number = 0
        self.innovations = []
        self.species = []
        self.genomes = []
        self.nodes = set()
        self.node_map = {}
        self.gene_map = {}
        self.input_nodes = set()
        self.output_nodes = set()
        if activations:
            NEAT.activations = activations

    def populate(self, population_size, num_inputs, num_outputs, output_activation=np.tanh):
        for pop_i in range(population_size):
            genome = Genome()
            self.genomes.append(genome)
            for output_i in range(num_outputs):
                for input_i in range(num_inputs):
                    gene, in_node, out_node = self.make_gene(input_i, num_inputs + output_i)
                    out_node.f_activation = output_activation
                    self.input_nodes.add(in_node.id)
                    self.output_nodes.add(out_node.id)
                    genome.add_gene(gene, in_node, out_node)

    def run(self, f_fitness, num_epochs=5000, report_every=10):
        epoch = 0
        best_fitness = None
        best_genome = None
        genome_fitness = defaultdict(lambda:-99999999)
        while epoch < num_epochs:
            epoch += 1
            this_generation = []
            for genome_i, genome in enumerate(self.genomes):
                new_genome = genome.clone()
                #new_genome.summary()
                new_genome.mutate(rate=0.5, p_sigma=1.0)
                if np.random.uniform() < 0.1:
                    if np.random.uniform() < 0.5:
                        new_genome.split_edge(self)
                    else:
                        new_genome.add_edge(self)
                nn = self.net_from_genome(new_genome)
                fitness = f_fitness(nn.eval)
                if best_fitness is None or best_fitness < fitness:
                    best_fitness = fitness
                    best_genome = new_genome
                if genome_fitness[genome_i] < fitness:
                    genome_fitness[genome_i] = fitness
                    this_genome = new_genome
                else:
                    this_genome = genome
                this_generation.append(this_genome)
            if epoch % report_every == 0:
                print(
                    'Epoch {} - fitness={}'.format(epoch, best_fitness),
                    'nodes={} edges={}'.format(
                        len(best_genome.nodes), len(best_genome.genes)
                    ))
                #new_genome.summary()
            self.genomes = this_generation
        return best_genome

    def make_node(self, id):
        if id in self.node_map:
            return self.node_map[id]
        node = self.node_class(id, f_activation=self.random_activation())
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

    def net_from_genome(self, genome):
        return NNGraph(
            genome.to_graph(), self.input_nodes, self.output_nodes)

    @property
    def innovation_number(self):
        return len(self.gene_map)

    @property
    def next_node_id(self):
        if self.node_map:
            return max(self.node_map) + 1
        return 0

    @classmethod
    def random_activation(self):
        return np.random.choice(self.activations)


if __name__ == '__main__':
    neat = NEAT()
