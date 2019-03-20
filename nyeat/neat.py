import copy
import random
from collections import defaultdict
from functools import reduce
from operator import itemgetter

import numpy as np

from nyeat.activations import gaussian, relu
from nyeat.genome import EdgeGene, Genome, NodeGene
from nyeat.nn_graph import NNGraph
from .species import Species


default_activations = (
    None, gaussian, relu, np.cos, np.sin, np.abs, np.tanh,)


class NEAT(object):
    activations = None

    def __init__(self, num_inputs, num_outputs, edge_class=EdgeGene, node_class=NodeGene,
            activations=None, output_activation=None):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.output_activation = output_activation
        self.edge_class = edge_class
        self.node_class = node_class
        self.next_innovation_number = 0
        self.innovations = []
        self.species_list = []
        self.genomes = {}
        self.node_map = {}
        self.gene_map = {}
        if not activations:
            activations = default_activations
        NEAT.activations = activations  # TODO: did I write this horseshit?

    def populate(self, population_size, p_connect=0.3):
        self.population_size = population_size
        for pop_i in range(population_size):
            genome = self.random_genome(p_connect=p_connect)
            self.genomes[genome.id] = genome
        self.species_list = Species.assign_species(self.genomes.values(), self.species_list)

    def random_genome(self, p_connect=0.3):
        num_inputs, num_outputs = self.num_inputs, self.num_outputs
        output_activation = self.output_activation
        if not isinstance(output_activation, (list, tuple)):
            output_activation = [output_activation] * num_outputs
        genome = Genome()
        self.genomes[genome.id] = genome
        for output_i in range(num_outputs):
            output_id = num_inputs + output_i
            out_node = self.make_node(output_id)
            if output_activation:
                out_node.f_activation = output_activation[output_i]
            genome.add_node(out_node, is_output=True)
            for input_id in range(num_inputs):
                in_node = self.make_node(input_id)
                genome.add_node(in_node, is_input=True)
                in_node.f_activation = None
                # potentially connect
                if np.random.uniform() > p_connect:
                    continue
                gene, in_node, out_node = self.make_gene(input_id, output_id)
                genome.add_gene(gene, in_node, out_node, is_input=True, is_output=True)
        return genome

    def breed(self, fitnesses, champions=0.05, cull=0.2,
        p_split=0.03, p_new_edge=0.3, p_mutate=0.8, p_sigma=0.08):
        #print('Breeding generation {}'.format(generation_i))
        next_generation = {}
        sorted_genomes = sorted([
                (f, i) for i, f in fitnesses.items()],
            reverse=True)
        sorted_genomes = list(map(lambda x: self.genomes[x[1]], sorted_genomes))
        # unchanged champions
        num_champions = int(champions * self.population_size)
        if num_champions:
            next_generation = {g.id: g for g in sorted_genomes[:num_champions]}
        # mutate better part of the population
        start_cull_at = int((1. - cull) * self.population_size)
        good_genomes = sorted_genomes[:start_cull_at]
        for genome in good_genomes:
            new_genome = genome.clone()
            new_genome.assign_new_id()
            if np.random.uniform() < p_split:
                new_genome.split_edge(self)
            if np.random.uniform() < p_new_edge:
                new_genome.add_edge(self)
            new_genome.mutate(rate=p_mutate, p_sigma=p_sigma)
            next_generation[new_genome.id] = new_genome
        # make children
        while len(next_generation) < self.population_size:
            best_parent, other_parent = random.sample(good_genomes, 2)
            if fitnesses[best_parent.id] < fitnesses[other_parent.id]:
                best_parent, other_parent = other_parent, best_parent
            new_genome = Genome.crossover(self, best_parent, other_parent)
            next_generation[new_genome.id] = new_genome
        self.genomes = next_generation
        return self.genomes

    def breed(self, fitnesses, champions=0.05, cull=0.2,
              p_split=0.03, p_new_edge=0.3, p_mutate=0.9, p_sigma=0.08,
              p_interspecies=0.001, p_adoption=0.1):
        next_generation = {}
        # remove stagnant species
        for species in self.species_list:
            species.update_stagnation(fitnesses)
            if species.is_stagnant:
                print('Species is stagnant')
        # cull and mutate population
        species_list = [s.clone() for s in self.species_list if not s.is_stagnant]
        new_species_list = [s.clone() for s in species_list]
        for species, new_species in zip(species_list, new_species_list):
            original_species_population = len(species.genomes)
            species.cull(fitnesses, ratio=cull)
            new_species.cull(fitnesses, ratio=(1 - champions))
            num_mutants = (original_species_population - len(new_species.genomes))
            for _ in range(num_mutants):
                genome = species.random_genome()
                new_genome = genome.clone()
                new_genome.assign_new_id()
                if np.random.uniform() < p_split:
                    new_genome.split_edge(self)
                if np.random.uniform() < p_new_edge:
                    new_genome.add_edge(self)
                new_genome.mutate(rate=p_mutate, p_sigma=p_sigma)
                new_species.add(new_genome)

        # fill in population with crossovers and randoms
        def updict(acc, i):
            acc.update(i)
            return acc
        next_generation = reduce(updict, [s.genomes for s in new_species_list], {})

        while len(next_generation) < self.population_size:
            product_of_interspecies_fornication = np.random.uniform() < p_interspecies
            if species_list:
                s_a_id, s_b_id = np.random.randint(0, len(species_list), 2, dtype=np.int32)
                s_a, s_a_new = species_list[s_a_id], new_species_list[s_a_id]
                # random interspecies breeding
                if product_of_interspecies_fornication:
                    s_b, s_b_new = species_list[s_b_id], new_species_list[s_b_id]
                else:
                    s_b, s_b_new = s_a, s_a_new
                g_a, g_b = s_a.random_genome(), s_b.random_genome()
                if fitnesses[g_a.id] < fitnesses[g_b.id]:
                    g_a, g_b = g_b, g_a
                new_genome = Genome.crossover(self, g_a, g_b)
            else:
                new_genome = self.random_genome()
            # adjust adoption rate https://apps.cs.utexas.edu/tech_reports/reports/tr/TR-1972.pdf
            if species_list and not product_of_interspecies_fornication and np.random.uniform() > p_adoption:
                s_a_new.add(new_genome)
            else:
                Species.assign_species([new_genome], new_species_list)
            next_generation[new_genome.id] = new_genome

        self.genomes = next_generation
        self.species_list = new_species_list
        self.genome_summary()

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
        gene.weight = np.random.normal(0, 1)
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
    def num_genomes(self):
        return len(self.genomes)

    @classmethod
    def random_activation(self):
        return np.random.choice(self.activations)

    def genome_summary(self):
        print(sum([len(s.genomes) for s in self.species_list]), 'genomes in',
              len(self.species_list), 'species')

    def species_fitness_summary(self):
        print('Species fitness')
        for s in self.species_list:
            print('age {}/{}'.format(s.age, s.stagnation_age),
                  'best fitness', s.best_fitness,
                  'genomes', len(s.genomes))

if __name__ == '__main__':
    neat = NEAT()
