#!/usr/bin/env python
"""Provides scikit interface."""

import networkx as nx
import random
import logging
logger = logging.getLogger(__name__)


class NeighborhoodEdgeSwap(object):

    def __init__(self, n_edges=1, n_neighbors=None, part_importance_estimator=None):
        self.part_importance_estimator = part_importance_estimator
        self.n_edges = n_edges
        self.n_neighbors = n_neighbors

    def fit(self, graphs, targets=None):
        return self

    def neighbors(self, graph):
        """neighbors."""
        if self.n_neighbors is None:
            return self.all_neighbors(graph)
        else:
            return self.sample_neighbors(graph)

    def all_neighbors(self, graph):
        """all_neighbors."""
        graphs = []
        edges = list(graph.edges())
        for j in range(len(edges) - 1):
            for k in range(j + 1, len(edges)):
                e1 = edges[j]
                e2 = edges[k]
                g = graph.copy()
                self._swap_edge(g, e1, e2)
                if nx.is_connected(g) and self._same_size(g, graph):
                    graphs.append(g)
        return graphs

    def sample_neighbors(self, graph):
        """sample_neighbors."""
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

    def swap(self, gg):
        """swap."""
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
