import pickle
from collections import defaultdict
from multiprocessing import Process, Queue

import numpy as np


def run_evaluator(
        evaluator_class, eval_args, eval_kwargs,
        work_queue, result_queue):
    evaluator = evaluator_class(*eval_args, **eval_kwargs)
    evaluator.run(work_queue, result_queue)


class Coordinator(object):
    def run(self, neat, num_workers, evaluator_class, eval_args=(),
            eval_kwargs={}, num_generations=100, report_every=5,
            save_file=None):
        work_queue = Queue()
        result_queue = Queue()
        print('Starting workers...')
        processes = []
        for worker_i in range(num_workers):
            should_render = worker_i == 0
            process = Process(
                target=run_evaluator,
                args=(
                    evaluator_class, eval_args, eval_kwargs, work_queue,
                    result_queue,))
            process.start()
            processes.append(process)
        print('Workers started.')

        generation_i = 0
        genome_fitness = defaultdict(lambda:-np.inf)
        while generation_i < num_generations:
            best_fitness = None
            best_genome = None
            generation_genome_fitness = {}
            generation_i += 1
            this_generation = neat.genomes
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

            save_top_n = int(len(neat.genomes) * 0.1)
            top_genomes = [(f, g) for g, f in generation_genome_fitness.items()][:save_top_n]
            top_genome_ids = set(g for _, g in top_genomes)
            #print('Evaluation complete.')
            next_generation = []
            for genome_i, (genome, new_genome) in enumerate(zip(neat.genomes, this_generation)):
                fitness = generation_genome_fitness[genome_i]
                # update best genome
                if best_fitness is None or best_fitness < fitness:
                    best_fitness = fitness
                    best_genome = new_genome
            # update population
            neat.genomes = self.breed_generation(
                neat, this_generation, generation_genome_fitness, genome_fitness)
            if generation_i % report_every == 0:
                print(
                    'Generation {} - fitness={}'.format(generation_i, best_fitness),
                    'nodes={} edges={}'.format(
                        len(best_genome.nodes), len(best_genome.genes)
                    ))
                if save_file:
                    with open(save_file, 'wb') as out_file:
                        pickle.dump(best_genome, out_file)
        # shutdown workers
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()
        return best_genome

    def breed_generation(self, neat, genomes, fitnesses, best_fitnesses):
        #print('Breeding generation {}'.format(generation_i))
        next_generation = []
        for genome_i, (genome, new_genome) in enumerate(zip(neat.genomes, genomes)):
            fitness = fitnesses[genome_i]
            if fitness > fitnesses[genome_i]: #in top_genome_ids: #genome_fitness[genome_i]:
                best_fitnesses[genome_i] = fitness
                genome = new_genome
            # create a mutant child
            new_genome = genome.clone()
            if np.random.uniform() < 0.05:
                new_genome.split_edge(neat)
            if np.random.uniform() < 0.15: #else:
                new_genome.add_edge(neat)
            new_genome.mutate(rate=0.8, p_sigma=1.0)
            next_generation.append(new_genome)
        return next_generation
