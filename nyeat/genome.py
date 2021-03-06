import copy
from collections import namedtuple

import networkx as nx
import numpy as np
import scipy

from nyeat import activations


class EdgeGene(object):
    def __init__(self, innovation, a, b, enabled=True, weight=1.):
        self.innovation = innovation
        self.enabled = enabled
        self.a = a
        self.b = b
        self.weight = weight

    def perturb(self, sigma=0.1):
        self.weight += np.random.normal(0., sigma)

    def clone(self):
        return copy.deepcopy(self)

    @property
    def nodes(self):
        return (self.a, self.b)


class NodeGene(object):
    activations = (
        None, activations.gaussian, activations.relu,
        np.cos, np.sin, np.abs, np.tanh,)

    def __init__(self, id, f_activation):
        self.id = id
        self.f_activation = f_activation


class Genome(object):
    def __init__(self):
        self.genes = {}
        self.edges = set()
        self.nodes = {}
        self.input_nodes = set()
        self.output_nodes = set()

    def add_gene(self, gene, in_node, out_node, is_input=False, is_output=False):
        self.genes[gene.innovation] = gene
        self.edges.add((in_node.id, out_node.id))
        for node in (in_node, out_node):
            self.nodes[node.id] = node
        if is_input:
            self.input_nodes.add(in_node.id)
        if is_output:
            self.output_nodes.add(out_node.id)

    def clone(self):
        return copy.deepcopy(self)

    def mutate(self, rate=0.5, p_sigma=0.5):
        for gene in self.genes.values():
            if np.random.uniform() < rate:
                gene.perturb(sigma=p_sigma)

    def split_edge(self, neat):
        target_gene = np.random.choice(self.enabled_genes)
        target_gene.enabled = False
        new_node = neat.make_node(neat.next_node_id)
        self.add_gene(*neat.make_gene(target_gene.a, new_node.id))
        self.add_gene(*neat.make_gene(new_node.id, target_gene.b))

    def add_edge(self, neat):
        """Attempts to add a new edge to the genome subject to
        the constraint that the graph remains acyclic. Returns
        True if a good edge was added.
        """
        num_nodes = len(self.nodes)
        max_edges = num_nodes * (num_nodes - 1)
        if len(self.genes) >= max_edges:
            print('Max edges of {} reached'.format(max_edges))
            return False
        new_g = self.to_graph()
        for node_i in self.nodes.values():
            for node_j in self.nodes.values():
                i, j = node_i.id, node_j.id
                if i == j or i in self.output_nodes or j in self.input_nodes:
                    continue
                key = (i, j)
                if key in self.edges:
                    continue
                g = new_g.copy()
                g.add_nodes_from(key)
                g.add_edge(i, j)
                try:
                    c = nx.find_cycle(g)
                except nx.exception.NetworkXNoCycle:
                    # No cycles? Good gene.
                    self.add_gene(*neat.make_gene(i, j))
                    return True
        return False

    def to_graph(self):
        g = nx.DiGraph()
        g.add_nodes_from((n.id, dict(n=n)) for n in self.nodes.values())
        g.add_edges_from(
            (g.a, g.b, dict(g=g)) for g in self.enabled_genes
        )
        return g

    def net_from_genome(self):
        return NNGraph(
            genome.to_graph(), self.input_nodes, self.output_nodes)

    @classmethod
    def crossover(cls, best_genome, other_genome):
        new_genome = Genome()
        for best_gene in best_genome.genes:
            if (best_gene.innovation in other_genome.genes
                    and np.random.uniform() < 0.5):
                chosen_gene = other_genome.genes[best_gene.innovation]
            else:
                chosen_gene = best_gene
            new_genome.add_gene(chosen_gene.clone())
        return new_genome

    @property
    def enabled_genes(self):
        return [g for g in self.genes.values() if g.enabled]

    def summary(self):
        print('Nodes\n--------')
        for node in self.nodes.values():
            if node.f_activation:
                name = node.f_activation.__name__
            else:
                name = 'none'
            print('{} {}'.format(node.id, name))
        print('Edges\n--------')
        for gene in self.genes.values():
            print('{} -> {}, w={}'.format(gene.a, gene.b, gene.weight))
