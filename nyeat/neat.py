import copy
from collections import defaultdict

import numpy as np

from nyeat.activations import gaussian, relu
from nyeat.genome import EdgeGene, Genome, NodeGene
from nyeat.nn_graph import NNGraph


default_activations = (
    None, gaussian, relu, np.cos, np.sin, np.abs, np.tanh,)


class NEAT(object):
    activations = None

    def __init__(self, edge_class=EdgeGene, node_class=NodeGene, activations=None):
        self.edge_class = edge_class
        self.node_class = node_class
        self.next_innovation_number = 0
        self.innovations = []
        self.species = []
        self.genomes = []
        self.node_map = {}
        self.gene_map = {}
        if not activations:
            activations = default_activations
        NEAT.activations = activations

    def populate(self, population_size, num_inputs, num_outputs, output_activation=np.tanh):
        for pop_i in range(population_size):
            genome = Genome()
            self.genomes.append(genome)
            for output_i in range(num_outputs):
                output_id = num_inputs + output_i
                for input_id in range(num_inputs):
                    gene, in_node, out_node = self.make_gene(input_id, output_id)
                    out_node.f_activation = output_activation
                    genome.add_gene(gene, in_node, out_node, is_input=True, is_output=True)

    # def run(self, f_fitness):
    #     generation_i = 0
    #     best_fitness = None
    #     best_genome = None
    #     genome_fitness = defaultdict(lambda:-np.inf)
    #     while generation_i < self.config.num_generations:
    #         generation_i += 1
    #         this_generation = []
    #         for genome_i, genome in enumerate(self.genomes):
    #             # create a mutant child
    #             new_genome = genome.clone()
    #             if np.random.uniform() < 0.05:
    #                 new_genome.split_edge(self)
    #             if np.random.uniform() < 0.15: #else:
    #                 new_genome.add_edge(self)
    #             new_genome.mutate(rate=0.8, p_sigma=5.0)
    #             # eval the new genome
    #             fitness = f_fitness(new_genome)
    #             if best_fitness is None or best_fitness < fitness:
    #                 best_fitness = fitness
    #                 best_genome = new_genome
    #             if genome_fitness[genome_i] < fitness:
    #                 genome_fitness[genome_i] = fitness
    #                 this_genome = new_genome
    #             else:
    #                 this_genome = genome
    #             this_generation.append(this_genome)
    #         if generation_i % self.config.report_every == 0:
    #             print(
    #                 'Generation {} - fitness={}'.format(generation_i, best_fitness),
    #                 'nodes={} edges={}'.format(
    #                     len(best_genome.nodes), len(best_genome.genes)
    #                 ))
    #         self.genomes = this_generation
    #     return best_genome

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
