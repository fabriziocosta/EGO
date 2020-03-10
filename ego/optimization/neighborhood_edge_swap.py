#!/usr/bin/env python
"""Provides scikit interface."""

import networkx as nx
import random
import logging
logger = logging.getLogger(__name__)


class NeighborhoodEdgeSwap(object):

    def __init__(self, n_edges=1, n_neighbors=10):
        self.n_edges = n_edges
        self.n_neighbors = n_neighbors

    def fit(self, graphs, targets):
        return self

    def neighbors(self, graph):
        if self.n_neighbors is None:
            return self._all_swaps(graph)
        neighs = [self.swap(graph) for i in range(self.n_neighbors)]
        return neighs

    def _swap_edge(self, g, e1, e2):
        s1, d1 = e1
        s2, d2 = e2
        ne1 = (s1, d2)
        ne2 = (s2, d1)
        args_e1 = g.edges[e1[0], e1[1]]
        args_e2 = g.edges[e2[0], e2[1]]
        g.remove_edge(*e1)
        g.remove_edge(*e2)
        g.add_edge(*ne1, **args_e1)
        g.add_edge(*ne2, **args_e2)

    def _swap(self, gg):
        g = gg.copy()
        es = [e for e in g.edges()]
        e1 = random.choice(es)
        e2 = random.choice(es)
        s1, d1 = e1
        s2, d2 = e2
        ne1 = (s1, d2)
        ne2 = (s2, d1)
        while s1 == s2 or s1 == d2 or d1 == s2 or d1 == d2 or s1 == d1 or s2 == d2 or g.has_edge(*ne1) or g.has_edge(*ne2):
            e1 = random.choice(es)
            e2 = random.choice(es)
            s1, d1 = e1
            s2, d2 = e2
            ne1 = (s1, d2)
            ne2 = (s2, d1)
        self._swap_edge(g, e1, e2)
        return g

    def _same_size(self, g, gg):
        return g.number_of_nodes() == gg.number_of_nodes() and g.number_of_edges() == gg.number_of_edges()

    def _all_swaps(self, gg):
        graphs = []
        edges = list(gg.edges())
        for j in range(len(edges) - 1):
            for k in range(j + 1, len(edges)):
                e1 = edges[j]
                e2 = edges[k]
                g = gg.copy()
                self._swap_edge(g, e1, e2)
                if nx.is_connected(g) and self._same_size(g, gg):
                    graphs.append(g)
        return graphs

    def swap(self, gg):
        g = gg.copy()
        for i in range(self.n_edges):
            gp = self._swap(g)
            guard = 0
            while nx.is_connected(gp) is False:
                gp = self._swap(g)
                guard += 1
                assert guard < 1e4, 'Too many failures'
            g = gp
        return g
