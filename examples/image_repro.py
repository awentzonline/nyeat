import pickle

import gym
import networkx as nx
import numpy as np
from PIL import Image
from scipy.misc import imresize, imsave

from nyeat import activations
from nyeat.cppn import CPPN
from nyeat.img_utils import arr_to_img
from nyeat.neat import Genome, NEAT
from nyeat.nn_graph import NNGraph
from nyeat.optimizer.coordinator import Coordinator
from nyeat.optimizer.evaluator import Evaluator
from nyeat.substrate import Substrate


def exhibition(config):
    evaluator = AtariEvaluator(config)
    while True:
        with open(config.file, 'rb') as in_file:
            genome = pickle.load(in_file)
        nn = NNGraph.from_genome(genome)
        cppn = CPPN(nn)
        # save images of the CPPN-generated weights
        visualize_cppn_weights(cppn)
        evaluator.evaluate(genome)


def save_cppn_image(cppn, size=(200, 200)):
    img_arr = cppn.render(size).transpose(1, 2, 0)
    img_arr *= 255.
    img = arr_to_img(img_arr, normalize=False)
    img.save('repro.png')


class ImageCopyEvaluator(Evaluator):
    def __init__(self, config):
        self.config = config
        self.img = Image.open(config.input_file).convert('RGB')
        self.img_arr = np.asarray(self.img).astype(np.float32)
        self.img_arr = self.img_arr / 255.

    def evaluate(self, genome):
        nn = NNGraph.from_genome(genome)
        cppn = CPPN(nn)
        repro = cppn.render(self.img.size)
        repro = repro.transpose(1, 2, 0)
        mse = np.sum(np.square(self.img_arr - repro))
        if np.random.uniform() < 0.01:
            save_cppn_image(cppn)
        return -mse


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('--file', default='genome.pkl')
    parser.add_argument('--exhibition', action='store_true')
    parser.add_argument('--pop-size', type=int, default=100)
    parser.add_argument('--hidden-size', type=int, default=100)
    parser.add_argument('--report-every', type=int, default=1)
    parser.add_argument('--num-generations', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=5)
    parser.add_argument('--resize', type=int, default=60)
    parser.add_argument('--env', default='Pong-v0')
    config = parser.parse_args()

    if config.file and config.exhibition:
        exhibition(config)
    else:
        neat = NEAT()
        num_channels = 3
        neat.populate(
            config.pop_size, 4, num_channels,
            output_activation=activations.sigmoid)

        coordinator = Coordinator()
        best_genome = coordinator.run(
            neat, config.num_workers, ImageCopyEvaluator,
            eval_args=(config,), num_generations=config.num_generations,
            report_every=config.report_every,
            save_file=config.file)
        with open(config.file, 'wb') as out_file:
            pickle.dump(best_genome, out_file)
        # show results
        best_genome.summary()  # this is the genome of the CPPN
        nn = NNGraph.from_genome(best_genome)
        cppn = CPPN(nn)
        # save images of the CPPN-generated weights
        save_cppn_image(cppn)
