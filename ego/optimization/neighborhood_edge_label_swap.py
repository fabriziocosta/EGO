#!/usr/bin/env python
"""Provides scikit interface."""

import random
import logging
logger = logging.getLogger(__name__)


class NeighborhoodEdgeLabelSwap(object):

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

    def sample_neighbors(self, graph):
        """sample_neighbors."""
        neighs = [self._swap_edges(graph, self.n_edges)
                  for i in range(self.n_neighbors)]
        return neighs

    def all_neighbors(self, graph):
        """all_neighbors."""
        graphs = [graph]
        for i in range(self.n_edges):
            graphs = [gg for g in graphs for gg in self._all_neighbors(g)]
        return graphs

    def _all_neighbors(self, graph):
        graphs = []
        edge_ids = list(graph.edges())
        for j in range(len(edge_ids) - 1):
            for k in range(j + 1, len(edge_ids)):
                g = graph.copy()
                self._swap_edge(g, edge_ids[j], edge_ids[k])
                graphs.append(g)
        return graphs

    def _swap_edge(self, gg, edge_id_j, edge_id_k):
        attr_j = gg.edges[edge_id_j].copy()
        attr_k = gg.edges[edge_id_k].copy()
        gg.edges[edge_id_j].update(attr_k)
        gg.edges[edge_id_k].update(attr_j)

    def _swap_edges(self, gg, n_edges):
        g = gg.copy()
        edge_ids = [edge_id for edge_id in g.edges()]
        random.shuffle(edge_ids)
        for i in range(n_edges):
            j = edge_ids[i * 2]
            k = edge_ids[i * 2 + 1]
            self._swap_edge(g, j, k)
        return g
