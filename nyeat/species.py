from itertools import zip_longest
import random

import numpy as np


class Species(object):
    next_species_id = 0

    def __init__(self, genomes, id=None, age=0, stagnation_time=10,
                 best_fitness=-np.inf, stagnation_age=None):
        if id is None:
            id = Species.assign_new_id()
        self.id = id
        if isinstance(genomes, (tuple, list)):
            genomes = {g.id: g for g in genomes}
        elif not isinstance(genomes, dict):
            genomes = {genomes.id: genomes}
        self.genomes = genomes
        self.age = age
        self.stagnation_age = stagnation_age or stagnation_time + age
        self.stagnation_time = stagnation_time
        self.best_fitness = best_fitness

    def distance(self, other):
        genome = list(self.genomes.values())[0]
        if genome is other:
            return 0.
        return genome.distance(other)

    def add(self, genome):
        self.genomes[genome.id] = genome

    def cull(self, fitnesses, ratio=0.2):
        sorted_genomes = self.sorted_genomes(fitnesses)
        start_cull_at = int((1. - ratio) * len(sorted_genomes))
        if start_cull_at == 0:
            return  # leave at least one
        best = sorted_genomes[:start_cull_at]
        self.genomes = {g.id: g for g in best}

    def sorted_genomes(self, fitnesses):
        sorted_genomes = sorted([
                (f, i) for i, f in fitnesses.items() if i in self.genomes],
            reverse=True)
        sorted_genomes = list(map(lambda x: self.genomes[x[1]], sorted_genomes))
        return sorted_genomes

    def update_stagnation(self, fitnesses):
        has_improved = False
        self.age += 1
        for genome in self.genomes.values():
            if not genome.id in fitnesses:
                print('****** Fitness not found ', genome.id, self.genomes, fitnesses)
            fitness = fitnesses[genome.id]
            if fitness > self.best_fitness and self.best_fitness != -np.inf:
                self.stagnation_age = self.age + self.stagnation_time
                self.best_fitness = fitness

    def normalize_fitnesses(self, fitnesses):
        normalized_fitnesses = {}
        for genome_id, fitness in fitnesses.items():
            if genome_id in self.genomes:
                normalized_fitnesses[genome_id] = fitness / float(len(self.genomes))
        return normalized_fitnesses

    def random_genome(self):
        return random.choice(list(self.genomes.values()))

    def clone(self, genomes=None):
        if genomes is None:
            genomes = dict(self.genomes)
        c = Species(genomes, id=self.id, age=self.age,
                    stagnation_age=self.stagnation_age,
                    stagnation_time=self.stagnation_time,
                    best_fitness=self.best_fitness)
        return c

    @property
    def is_empty(self):
        return len(genomes) == 0

    @classmethod
    def assign_new_id(cls):
        id = Species.next_species_id
        Species.next_species_id += 1
        return id

    @property
    def is_stagnant(self):
        return self.stagnation_age <= self.age

    @classmethod
    def assign_species(cls, genomes, species_list, member_threshold=4.):
        for genome in genomes:
            found = False
            for species in species_list:
                #print(species.distance(genome))
                if species.distance(genome) <= member_threshold:
                    species.add(genome)
                    found = True
                    break
            # create a new species if no match is found
            if not found:
                species = cls(genome)
                species_list.append(species)
        return species_list
