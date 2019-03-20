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
        self.update_representative()
        self.age = age
        self.stagnation_age = stagnation_age or stagnation_time + age
        self.stagnation_time = stagnation_time
        self.best_fitness = best_fitness

    def distance(self, other):
        if self.representative is other:
            return 0.
        return self.representative.distance(other)

    def clear_genomes(self):
        self.genomes = {}
        return self

    def add(self, genome):
        self.genomes[genome.id] = genome

    def cull(self, fitnesses, ratio=0.2, leave=0):
        sorted_genomes = self.sorted_genomes(fitnesses)
        start_cull_at = int((1. - ratio) * len(sorted_genomes))
        start_cull_at = max(leave, start_cull_at)
        # if start_cull_at == 0:
        #     return  # leave at least one
        best = sorted_genomes[:start_cull_at]
        self.genomes = {g.id: g for g in best}
        return self

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
            fitness = fitnesses[genome.id]
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                if self.best_fitness != -np.inf:
                    self.stagnation_age = self.age + self.stagnation_time

    def get_champions(self, fitnesses, ratio):
        sorted_genomes = self.sorted_genomes()
        last_champion = int(ratio * len(sorted_genomes))
        return self.genomes.values()[:last_champion]

    def normalized_fitnesses(self, fitnesses):
        normalized_fitnesses = {}
        num_genomes = float(len(self.genomes))
        for genome_id, fitness in fitnesses.items():
            if genome_id in self.genomes:
                normalized_fitnesses[genome_id] = fitness / num_genomes
        return normalized_fitnesses

    # def normalize_fitnesses(self, fitnesses):
    #     normalized_fitnesses = {}
    #     for genome in self.genomes:
    #         fitness = fitnesses[genome.id]
    #         normalized_fitnesses[genome.id] = fitness / float(len(self.genomes))
    #     sum_of_fitness = sum(normalized_fitnesses.values())
    #     for genome_id in normalized_fitnesses:
    #         normalized_fitnesses[genome_id] /= sum_of_fitness
    #     return normalized_fitnesses, sum_of_fitness

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
    def representative(self):
        return self._representative

    def update_representative(self):
        self._representative = random.choice(list(self.genomes.values()))

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

    @property
    def population(self):
        return len(self.genomes)
    @classmethod
    def assign_species(cls, genomes, species_list, last_species_list, member_threshold=3.):
        # put each genome in their best species or make a new one
        distances = []
        for genome in genomes:
            best_species = None
            best_distance = np.inf
            for species in species_list:
                target_species = species#last_species if last_species else species
                distance = target_species.distance(genome)
                distances.append(distance)
                if distance <= member_threshold and distance < best_distance:
                    best_distance = distance
                    best_species = species

            if best_species:
                best_species.add(genome)
            else:
                species = cls([genome])
                species_list.append(species)
        # update
        for species in species_list:
            species.update_representative()
        print('Mean/std genome distance', np.mean(distances), np.std(distances))
        return species_list
