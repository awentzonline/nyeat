from collections import defaultdict
from multiprocessing import Process, Queue

import numpy as np


def run_evaluator(evaluator_class, work_queue, result_queue):
    evaluator = evaluator_class()
    evaluator.run(work_queue, result_queue)


class Coordinator(object):
    def run(self, neat, num_workers, evaluator_class, num_generations=100, report_every=5):
        work_queue = Queue()
        result_queue = Queue()
        print('Starting workers...')
        processes = []
        for _ in range(num_workers):
            process = Process(
                target=run_evaluator, args=(evaluator_class, work_queue, result_queue))
            process.start()
            processes.append(process)
        print('Workers started.')

        generation_i = 0
        best_fitness = None
        best_genome = None
        genome_fitness = defaultdict(lambda:-np.inf)
        while generation_i < num_generations:
            generation_i += 1
            this_generation = []
            generation_genome_fitness = {}
            #print('Breeding generation {}'.format(generation_i))
            for genome in neat.genomes:
                # create a mutant child
                new_genome = genome.clone()
                if np.random.uniform() < 0.05:
                    new_genome.split_edge(neat)
                if np.random.uniform() < 0.15: #else:
                    new_genome.add_edge(neat)
                new_genome.mutate(rate=0.8, p_sigma=5.0)
                this_generation.append(new_genome)

            #print('Enqueuing...')
            for genome_i, genome in enumerate(this_generation):
                work_queue.put((genome_i, genome))

            #print('Working generation {}'.format(generation_i))
            num_results_pending = len(neat.genomes)
            while True:
                genome_i, fitness = result_queue.get()
                generation_genome_fitness[genome_i] = fitness
                num_results_pending -= 1
                if num_results_pending <= 0:
                    break

            #print('Evaluation complete.')
            next_generation = []
            for genome_i, (genome, new_genome) in enumerate(zip(neat.genomes, this_generation)):
                fitness = generation_genome_fitness[genome_i]
                # update best genome
                if best_fitness is None or best_fitness < fitness:
                    best_fitness = fitness
                    best_genome = new_genome
                # update population
                if fitness > genome_fitness[genome_i]:
                    genome_fitness[genome_i] = fitness
                    this_genome = new_genome
                else:
                    this_genome = genome
                next_generation.append(this_genome)
            neat.genomes = next_generation

            if generation_i % report_every == 0:
                print(
                    'Generation {} - fitness={}'.format(generation_i, best_fitness),
                    'nodes={} edges={}'.format(
                        len(best_genome.nodes), len(best_genome.genes)
                    ))
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()
        return best_genome
