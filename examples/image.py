import networkx as nx
import numpy as np

from nyeat import activations
from nyeat.cppn import CPPN
from nyeat.img_utils import arr_to_img
from nyeat.neat import Genome, NEAT
from nyeat.nn_graph import NNGraph


pop_size, input_dims, output_dims = 1, 4, 3
neat = NEAT()
neat.populate(
    pop_size, input_dims, output_dims,
    output_activation=activations.sigmoid)

genome = neat.genomes[0]
# do some mutations
for i in range(300):
    if i % 10 == 0:
        print('mutating...', i)
    if np.random.uniform() < 0.5:
        genome.split_edge(neat)
    else:
        genome.add_edge(neat)
# create a CPPN from this genome and render an image
nn = NNGraph.from_genome(genome)
cppn = CPPN(nn)
img_shape = (512, 200)
img_arr = cppn.render(img_shape, ((-1, 1), (-1, 1), (-1, 1)))
img_arr = img_arr * 255.
img_arr = img_arr.transpose(1, 2, 0)
img = arr_to_img(img_arr, normalize=False)
img.save('nyeat.png')
