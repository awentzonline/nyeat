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
    def __init__(self):
        self.worker_processes = []
        self.work_queue = Queue()
        self.result_queue = Queue()

    def run(self, neat, num_workers, evaluator_class, eval_args=(),
            eval_kwargs={}, num_generations=100, report_every=5,
            save_file=None):
        print('Starting workers...')
        self.start_workers(num_workers, evaluator_class, eval_args, eval_kwargs)
        print('{} workers started.'.format(num_workers))

        generation_i = 0
        fitnesses = defaultdict(lambda:-np.inf)
        while generation_i < num_generations:
            best_fitness = None
            best_genome = None
            genome_fitnesses = {}
            generation_i += 1
            #print('Enqueuing...')
            for genome_i, genome in neat.enumerated_genomes:
                self.work_queue.put((genome_i, genome))

            #print('Working generation {}'.format(generation_i))
            num_results_pending = neat.num_genomes
            while True:
                genome_i, fitness = self.result_queue.get()
                genome_fitnesses[genome_i] = fitness
                num_results_pending -= 1
                if num_results_pending <= 0:
                    break

            # save the current best for display
            next_generation = []
            for genome_i, new_genome in neat.enumerated_genomes:
                fitness = genome_fitnesses[genome_i]
                # update best genome
                if best_fitness is None or best_fitness <= fitness:
                    best_fitness = fitness
                    best_genome = new_genome

            # update population
            neat.breed(genome_fitnesses)

            # save the best model and show some feedback
            if generation_i % report_every == 0:
                print(
                    'Generation {} - fitness={}'.format(generation_i, best_fitness),
                    'nodes={} edges={}'.format(
                        len(best_genome.nodes), len(best_genome.genes)
                    ))
                if save_file:
                    with open(save_file, 'wb') as out_file:
                        pickle.dump(best_genome, out_file)
        self.stop_workers()
        return best_genome

    def start_workers(
            self, num_workers, evaluator_class, eval_args, eval_kwargs):
        for worker_i in range(num_workers):
            should_render = worker_i == 0
            process = Process(
                target=run_evaluator,
                args=(
                    evaluator_class, eval_args, eval_kwargs, self.work_queue,
                    self.result_queue,))
            process.start()
            self.worker_processes.append(process)

    def stop_workers(self):
        # shutdown workers
        for process in self.worker_processes:
            process.terminate()
        for process in self.worker_processes:
            process.join()
        self.worker_processes = []
