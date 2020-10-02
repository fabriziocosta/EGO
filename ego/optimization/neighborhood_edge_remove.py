#!/usr/bin/env python
"""Provides scikit interface."""

import networkx as nx
import random
from itertools import combinations
import logging
logger = logging.getLogger(__name__)


class NeighborhoodEdgeRemove(object):

    def __init__(self, n_edges=1, n_neighbors=10, part_importance_estimator=None):
        self.part_importance_estimator = part_importance_estimator
        self.n_neighbors = n_neighbors
        self.n_edges = n_edges

    def fit(self, graphs, targets):
        """fit."""
        return self

    def neighbors(self, graph):
        """neighbors."""
        if self.n_neighbors is None:
            return self.all_neighbors(graph)
        else:
            return self.sample_neighbors(graph)

    def all_neighbors(self, graph):
        """all_neighbors."""
        edges = list(graph.edges())
        combs = combinations(edges, self.n_edges)
        neighs = [self._remove_edge(graph, edge_ids) for edge_ids in combs]
        return neighs

    def sample_neighbors(self, graph):
        """sample_neighbors."""
        neighs = [self._remove(graph, self.n_edges)
                  for i in range(self.n_neighbors)]
        return neighs

    def _remove_edge(self, gg, edge_ids):
        g = gg.copy()
        for edge_id in edge_ids:
            if edge_id in set(g.edges()) and len(g.edges()) > 1:
                u, v = edge_id
                g.remove_edge(u, v)
                max_cc = max(nx.connected_components(g), key=lambda x: len(x))
                g = nx.subgraph(g, max_cc).copy()
        return g

    def _remove(self, gg, n_edges):
        g = gg.copy()
        edge_ids = [edge_id for edge_id in g.edges()]
        random.shuffle(edge_ids)
        edge_ids = edge_ids[:n_edges]
        g = self._remove_edge(g, edge_ids)
        return g
