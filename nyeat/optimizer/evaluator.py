import numpy as np


class Evaluator(object):
    """Worker which evaluates the fitness of a genome."""
    def run(self, work_queue, result_queue):
        while True:
            genome_i, genome = work_queue.get()
            fitness = self.evaluate(genome)
            result_queue.put((genome_i, fitness))

    def evalute(self, genome):
        pass
