import networkx as nx
import numpy as np

from nyeat.cppn import CPPN
from nyeat.img_utils import arr_to_img
from nyeat.neat import Genome, NEAT
from nyeat.nn_graph import NNGraph


pop_size, input_dims, output_dims = 1, 4, 3
neat = NEAT()
neat.populate(pop_size, input_dims, output_dims)

genome = neat.genomes[0]
# do some mutations
for i in range(300):
    if i % 10 == 0:
        print("mutating...", i)
    if np.random.uniform() < 0.5:
        genome.split_edge(neat)
    else:
        genome.add_edge(neat)
# create a CPPN from this genome and render an image
graph = genome.to_graph()
input_nodes = [0, 1, 2, 3]
output_nodes = [4, 5, 6]
nn = NNGraph(graph, input_nodes, output_nodes)
cppn = CPPN(nn)
grid_shape = (512, 200)
img_arr = cppn.render(grid_shape, ((-1, 1), (-1, 1), (-1, 1)))
img = arr_to_img(img_arr, normalize=True)
img.save('nyeat.png')
