import networkx as nx
import numpy as np

from nyeat import activations
from nyeat.neat import Genome, NEAT
from nyeat.nn_graph import NNGraph


neat = NEAT(activations=(np.tanh,))
neat.populate(100, 3, 1, output_activation=activations.sigmoid)

xor_values = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
]

def eval_f(f, verbose=False):
    err = 0.
    for x, y, z in xor_values:
        inputs = np.array([x, y, 1.]).astype(np.float32)
        p_z = f(inputs)[0]
        this_err = np.abs(z - p_z)
        if verbose:
            print(x, y, p_z, this_err)
        err += this_err
    return -(err ** 2)

best_genome = neat.run(eval_f, num_epochs=400)
nn = neat.net_from_genome(best_genome)
for x, y, z in xor_values:
    inputs = np.array([x, y, 1.]).astype(np.float32)
    p_z = nn.eval(inputs)[0]
    print(inputs, p_z, z)
eval_f(nn.eval, verbose=True)
best_genome.summary()
