import copy
import random
from collections import defaultdict
from operator import itemgetter

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
        self.population_size = population_size
        for pop_i in range(population_size):
            genome = Genome()
            self.genomes.append(genome)
            for output_i in range(num_outputs):
                output_id = num_inputs + output_i
                for input_id in range(num_inputs):
                    gene, in_node, out_node = self.make_gene(input_id, output_id)
                    out_node.f_activation = output_activation
                    genome.add_gene(gene, in_node, out_node, is_input=True, is_output=True)
                    #genome.summary()

    def breed(self, fitnesses, champions=0.05, cull=0.2, state={'n': 0}):
        #print('Breeding generation {}'.format(generation_i))
        next_generation = []
        sorted_genomes = sorted([
                (f, i) for i, f in fitnesses.items()],
            reverse=True)
        sorted_genomes = list(map(lambda x: self.genomes[x[1]], sorted_genomes))
        # unchanged champions
        num_champions = int(champions * self.population_size)
        if num_champions:
            next_generation = sorted_genomes[:num_champions]
        # mutated better part of the population
        start_cull_at = int((1. - cull) * self.population_size)
        good_genomes = sorted_genomes[:start_cull_at]
        for genome in good_genomes:
            new_genome = genome.clone()
            if np.random.uniform() < 0.05:
                new_genome.split_edge(self)
            if np.random.uniform() < 0.15: #else:
                new_genome.add_edge(self)
            new_genome.mutate(rate=0.8, p_sigma=0.2)
            next_generation.append(new_genome)
        # make children
        while len(next_generation) < self.population_size:
            # new_genome = random.choice(good_genomes).clone()
            # if np.random.uniform() < 0.05:
            #     new_genome.split_edge(self)
            # if np.random.uniform() < 0.15:
            #     new_genome.add_edge(self)
            # new_genome.mutate(rate=0.8, p_sigma=1.0)
            good_id, other_id = np.random.randint(0, len(good_genomes), 2)
            if fitnesses[good_id] < fitnesses[other_id]:
                good_id, other_id = other_id, good_id
            best_parent, other_parent = good_genomes[good_id], good_genomes[other_id]
            new_genome = Genome.crossover(self, best_parent, other_parent)
            next_generation.append(new_genome)
        self.genomes = next_generation
        return self.genomes

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

    @property
    def enumerated_genomes(self):
        return enumerate(self.genomes)

    @property
    def num_genomes(self):
        return len(self.genomes)

    @classmethod
    def random_activation(self):
        return np.random.choice(self.activations)


class Species(object):
    def __init__(self, genome):
        self.genome = genome
        self.genome_ids = {}

    def distance(self, other):
        if genome == other:
            return 0.
        return self.genome.difference(other)

    def add(self, genome_id):
        self.genome_ids[genome_id] = True

    def normalize_fitnesses(self, fitnesses):
        for genome_id, fitness in fitnesses.items():
            if genome_id in self.genomes_ids:
                fitnesses[genome_id] = fitness / float(len(self.genome_ids))

    @classmethod
    def speciate(cls, genomes, species_list, max_distance=5):
        for genome_id, genome in enumerate(genomes):
            found = False
            for species in species_list:
                if species.distance(genome) < max_distance:
                    species.add(genome_id)
                    found = True
                    break
            # create a new species if no match is found
            if not found:
                species = cls(genome)



if __name__ == '__main__':
    neat = NEAT()
