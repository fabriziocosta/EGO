#!/usr/bin/env python
"""Provides scikit interface."""

import random
import logging
logger = logging.getLogger(__name__)


class NeighborhoodNodeLabelSwap(object):

    def __init__(self, n_nodes=1, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.n_nodes = n_nodes

    def fit(self, graphs, targets):
        """fit."""
        return self

    def neighbors(self, graph):
        """neighbors."""
        if self.n_neighbors is None:
            graphs = [graph]
            for i in range(self.n_nodes):
                graphs = [gg for g in graphs for gg in self._all_neighbors(g)]
            return graphs

        neighs = [self._swap_nodes(graph, self.n_nodes)
                  for i in range(self.n_neighbors)]
        return neighs

    def _all_neighbors(self, graph):
        graphs = []
        node_ids = list(graph.nodes())
        for j in range(len(node_ids) - 1):
            for k in range(j + 1, len(node_ids)):
                g = graph.copy()
                self._swap_node(g, node_ids[j], node_ids[k])
                graphs.append(g)
        return graphs

    def _swap_node(self, gg, node_id_j, node_id_k):
        attr_j = gg.nodes[node_id_j].copy()
        attr_k = gg.nodes[node_id_k].copy()
        gg.nodes[node_id_j].update(attr_k)
        gg.nodes[node_id_k].update(attr_j)

    def _swap_nodes(self, gg, n_nodes):
        g = gg.copy()
        node_ids = [node_id for node_id in g.nodes()]
        random.shuffle(node_ids)
        for i in range(n_nodes):
            j = node_ids[i * 2]
            k = node_ids[i * 2 + 1]
            self._swap_node(g, j, k)
        return g
