import pickle

import gym
import networkx as nx
import numpy as np
from scipy.misc import imresize, imsave

from nyeat import activations
from nyeat.cppn import CPPN
from nyeat.img_utils import arr_to_img
from nyeat.neat import Genome, NEAT
from nyeat.nn_graph import NNGraph
from nyeat.optimizer.coordinator import Coordinator
from nyeat.optimizer.evaluator import Evaluator
from nyeat.substrate import (
    DCTSubstrate, NeuralNetwork1DSubstrate, NeuralNetwork2DSubstrate, Substrate
)


def exhibition(config):
    evaluator = AtariEvaluator(config)
    while True:
        with open(config.file, 'rb') as in_file:
            genome = pickle.load(in_file)
        genome.plot_graph('genome_graph.png')
        genome.summary()
        nn = NNGraph.from_genome(genome)
        cppn = CPPN(nn, use_bias=config.use_bias, use_radius=config.use_radius)
        # save images of the CPPN-generated weights
        visualize_cppn_weights(cppn)
        evaluator.evaluate(genome)


def visualize_cppn_weights(cppn, substrate=None):
    from scipy.fftpack import idct
    output_shape = (20, 128, 128)
    #output_shape = (128, 128)
    #ranges = ((2 * np.pi, 0.),) * len(output_shape)
    values = cppn.render(output_shape)#, ranges=ranges)
    #print(values.shape)
    for i, img_arr in enumerate(values):
        #print(img_arr.shape)
        #img_arr = img_arr.transpose(1, 2, 0)
        img_arr = img_arr.reshape((20 * 128, 128))
        img = arr_to_img(img_arr, normalize=True)
        img.save('nyeat_{}.png'.format(i))
        # dct_img_arr = idct(img_arr)
        # img = arr_to_img(dct_img_arr, normalize=True)
        # img.save('nyeat_dct_{}.png'.format(i))
    if substrate:
        for i, matrices in enumerate(zip(substrate.matrices, substrate.biases)):
            #img_arr = img_arr.T
            for j, img_arr in enumerate(matrices):
                # if i == j == 0:
                #     print(img_arr)
                if len(img_arr.shape) > 2:
                    img_arr = img_arr.reshape(img_arr.shape[0], np.prod(img_arr.shape[1:]))
                if len(img_arr.shape) == 1:
                    img_arr = img_arr[..., None]
                img = arr_to_img(img_arr, normalize=True)
                img.save('nyeat_mat_{}_{}.png'.format(i, j))


class AtariEvaluator(Evaluator):
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env)
        self.observation_size = (config.observation_size, config.observation_size)
        self.num_hidden = config.hidden_size
        self.num_actions = self.env.action_space.n
        # self.substrate = NeuralNetwork1DSubstrate(
        #     np.prod(self.observation_size),
        #     ((self.num_hidden, activations.relu),#np.tanh),
        #      (self.num_actions, None)))
        self.substrate = NeuralNetwork2DSubstrate(
            self.observation_size,
            (
                ((self.num_hidden,), activations.relu),#np.tanh),
                ((self.num_actions,), None)
            )
        )

    def evaluate(self, genome):
        nn = NNGraph.from_genome(genome)
        cppn = CPPN(nn, use_bias=self.config.use_bias, use_radius=self.config.use_radius)
        self.substrate.update_weights(cppn)
        # for matrix in self.substrate.matrices:
        #     print(matrix.min(), matrix.mean(), matrix.max())

        # visualize weights
        if np.random.uniform() < 0.91:
            visualize_cppn_weights(cppn, self.substrate)
        return self.run_episode()

    def run_episode(self):
        total_reward = 0.
        observation = self.env.reset()
        done = False
        # random start
        for i in range(np.random.randint(0, 40)):
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            # if self.config.exhibition:
            #     self.env.render()
        # run the rest of the episode
        while not done:
            if self.config.exhibition:
                self.env.render()
            observation = self.process_observation(observation)
            #print(observation.min(), observation.mean(), observation.max())
            actions = self.substrate.eval(observation.transpose((1, 2, 0)))
            action = np.argmax(actions)
            #print(actions.min(), actions.mean(), actions.max())
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
        #print('episode complete', total_reward)
        return total_reward

    def process_observation(self, img):
        img = imresize(
            img.astype(np.float32), self.observation_size,
            interp='bicubic')
        # to single channel
        img = np.mean(img, axis=2, keepdims=True).transpose(2, 0, 1)
        if False and np.random.uniform() < 0.01:
            imsave('obs.png', img[0].astype(np.uint8))
        img /= 255.
        return img


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='genome.pkl')
    parser.add_argument('--exhibition', action='store_true')
    parser.add_argument('--pop-size', type=int, default=100)
    parser.add_argument('--hidden-size', type=int, default=100)
    parser.add_argument('--report-every', type=int, default=1)
    parser.add_argument('--num-generations', type=int, default=100000)
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--observation-size', type=int, default=60)
    parser.add_argument('--env', default='SpaceInvaders-v4')
    parser.add_argument('--use-bias', action='store_true')
    parser.add_argument('--use-radius', action='store_true')
    config = parser.parse_args()

    np.random.seed(1337)

    if config.file and config.exhibition:
        exhibition(config)
    else:
        num_layers = 2
        outputs_per_layer = 2
        num_inputs = config.use_bias + config.use_radius + 3
        neat = NEAT(num_inputs, num_layers * outputs_per_layer, output_activation=None)
        neat.populate(config.pop_size)

        num_workers = config.num_workers
        coordinator = Coordinator()
        best_genome = coordinator.run(
            neat, num_workers, AtariEvaluator,
            eval_args=(config,), num_generations=config.num_generations,
            report_every=config.report_every,
            save_file=config.file)
        if best_genome:
            with open(config.file, 'wb') as out_file:
                pickle.dump(best_genome, out_file)
            # show results
            best_genome.summary()  # this is the genome of the CPPN
            best_genome.plot_graph('best_genome.png')
            nn = NNGraph.from_genome(best_genome)
            cppn = CPPN(nn)
            # save images of the CPPN-generated weights
            visualize_cppn_weights(cppn)
