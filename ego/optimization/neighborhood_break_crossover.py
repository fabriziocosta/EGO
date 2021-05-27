#!/usr/bin/env python
"""Provides scikit interface."""

import networkx as nx
import numpy as np
import random
import logging
from ego.decomposition.dbreak import decompose_break
from ego.vectorize import set_feature_size
from ego.encode import make_encoder

logger = logging.getLogger(__name__)


def make_size_filter(graphs, z=2):
    lens = [len(g) for g in graphs]
    mu, sigma = np.mean(lens), np.std(lens)

    def size_filter(g):
        if z is None:
            return True
        if mu - z * sigma < len(g) < mu + z * sigma:
            return True
        else:
            return False
    return size_filter


def filter_by_size(graphs, size_filter=None):
    return [g for g in graphs if size_filter(g)]


class NeighborhoodBreakCrossover(object):

    def __init__(self, n_edges=1, min_size=5, max_size=.5, n_neighbors=None, tournament_size_factor=10, size_n_std_to_accept=None):
        self.size_n_std_to_accept = size_n_std_to_accept
        self.tournament_size_factor = tournament_size_factor
        self.n_edges = n_edges
        self.min_size = min_size
        self.max_size = max_size
        self.n_neighbors = n_neighbors

    def fit(self, graphs, targets=None):
        """fit."""
        self.edge_labels = list(set([g.edges[e]['label'] for g in graphs for e in g.edges()]))
        self.size_filter = make_size_filter(graphs, z=self.size_n_std_to_accept)
        if targets is None:
            self.graphs = graphs[:]
            random.shuffle(self.graphs)
        else:
            # store graphs sorted by target
            ids = sorted(range(len(graphs)), key=lambda i: targets[i], reverse=True)
            self.graphs = [graphs[id].copy() for id in ids]
        return self

    def neighbors(self, graph):
        """neighbors."""
        if self.n_neighbors is None:
            n = len(self.graphs)
        else:
            n = min(len(self.graphs), self.n_neighbors)
        neighs = []
        m = self.tournament_size_factor * n
        m = min(m, len(self.graphs))
        # select the best n out of m random graphs
        # sample m integers from 0 to len(graphs)
        ids = random.sample(range(len(self.graphs)), m)
        # sort them and select the smallest n
        sorted_ids = sorted(ids)[:n]
        for i in sorted_ids:
            candidate_graphs = self._compose(graph, self.graphs[i])
            neighs.extend(filter_by_size(candidate_graphs, size_filter=self.size_filter))
        return neighs

    def _compose(self, ga, gb):
        feature_size, bitmask = set_feature_size(nbits=14)
        df = decompose_break(min_size=self.min_size, max_size=self.max_size, n_edges=self.n_edges)
        encoding_func = make_encoder(df, preprocessors=None, bitmask=bitmask, seed=1)
        codes_a, fragments_a = encoding_func(ga)
        codes_b, fragments_b = encoding_func(gb)
        gs = []
        for fragment_a in fragments_a:
            fragment_a = nx.convert_node_labels_to_integers(fragment_a)
            for fragment_b in fragments_b:
                #choose a node in a and one in b and join them with an edge
                fragment_b = nx.convert_node_labels_to_integers(fragment_b)
                ia = random.choice(range(len(fragment_a)))
                jb = random.choice(range(len(fragment_b)))+len(fragment_a)
                g0 = nx.disjoint_union(fragment_a, fragment_b)
                assert(jb in list(g0.nodes()))
                g0.add_edge(ia,jb, label=random.choice(self.edge_labels))

                gs.append(g0.copy())
        return gs
