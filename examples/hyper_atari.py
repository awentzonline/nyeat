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



def visualize_cppn_weights(cppn):
    values = cppn.render((200, 200))
    for i, img_arr in enumerate(values):
        #img_arr = (img_arr + 1) * 127.
        img_arr = img_arr.T
        img = arr_to_img(img_arr, normalize=True)
        img.save('nyeat_{}.png'.format(i))


class AtariEvaluator(Evaluator):
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env)
        self.observation_size = (config.observation_size, config.observation_size)
        self.num_hidden = config.hidden_size
        self.num_actions = self.env.action_space.n
        self.substrate = Substrate(
            np.prod(self.observation_size),
            ((self.num_hidden, np.tanh),
             (self.num_actions, activations.softmax)))

    def evaluate(self, genome):
        nn = NNGraph.from_genome(genome)
        cppn = CPPN(nn)
        self.substrate.update_weights(cppn)
        # for matrix in self.substrate.matrices:
        #     print(matrix.min(), matrix.mean(), matrix.max())

        # visualize weights
        if np.random.uniform() < 0.1:
            visualize_cppn_weights(cppn)
        return self.run_episode()

    def run_episode(self):
        total_reward = 0.
        observation = self.env.reset()
        done = False
        while not done:
            if self.config.exhibition:
                self.env.render()
            observation = self.process_observation(observation)
            #print(observation.min(), observation.mean(), observation.max())
            actions = self.substrate.eval(observation.ravel())
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
    parser.add_argument('--num_generations', type=int, default=50)
    parser.add_argument('--observation-size', type=int, default=60)
    parser.add_argument('--env', default='Pong-v0')
    config = parser.parse_args()

    if config.file and config.exhibition:
        exhibition(config)
    else:
        neat = NEAT()
        num_layers = 2
        neat.populate(
            config.pop_size, 4, num_layers, output_activation=None)

        num_workers = 5
        coordinator = Coordinator()
        best_genome = coordinator.run(
            neat, num_workers, AtariEvaluator,
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
        visualize_cppn_weights(cppn)
