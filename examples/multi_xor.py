import networkx as nx
import numpy as np

from nyeat import activations
from nyeat.cppn import CPPN
from nyeat.img_utils import arr_to_img
from nyeat.neat import Genome, NEAT
from nyeat.nn_graph import NNGraph
from nyeat.optimizer.coordinator import Coordinator
from nyeat.optimizer.evaluator import Evaluator
from nyeat.substrate import Substrate


neat = NEAT(activations=(np.tanh,))
neat.populate(100, 3, 1, output_activation=activations.sigmoid)

xor_values = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
]


def exhibition(nn):
    for x, y, z in xor_values:
        inputs = np.array([x, y]).astype(np.float32)
        p_z = nn.eval(inputs)[0]
        print(inputs, p_z, z)
    print('\n')


def visualize_cppn_weights(cppn):
    values = cppn.render((200, 200))
    for i, img_arr in enumerate(values):
        #img_arr = (img_arr + 1) * 127.
        img_arr = img_arr.T
        img = arr_to_img(img_arr, normalize=True)
        img.save('nyeat_{}.png'.format(i))


class XOREvaluator(Evaluator):
    def evaluate(self, genome):
        nn = NNGraph.from_genome(genome)
        err = 0.
        for x, y, z in xor_values:
            inputs = np.array([x, y, 1.]).astype(np.float32)
            p_z = nn.eval(inputs)[0]
            this_err = np.abs(z - p_z)
            err += this_err
        return -(err ** 2)


num_workers = 5
coordinator = Coordinator()
best_genome = coordinator.run(
    neat, num_workers, XOREvaluator, num_generations=150)
# show results
best_genome.summary()  # this is the genome of the CPPN
nn = NNGraph.from_genome(best_genome)
exhibition(nn)
