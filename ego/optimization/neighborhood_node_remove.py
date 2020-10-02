#!/usr/bin/env python
"""Provides scikit interface."""

import networkx as nx
import random
from itertools import combinations
import logging
logger = logging.getLogger(__name__)


class NeighborhoodNodeRemove(object):

    def __init__(self, n_nodes=1, n_neighbors=10, part_importance_estimator=None):
        self.part_importance_estimator = part_importance_estimator
        self.n_neighbors = n_neighbors
        self.n_nodes = n_nodes

    def fit(self, graphs, targets):
        """fit."""
        return self

    def neighbors(self, graph):
        """neighbors."""
        if self.n_neighbors is None:
            nodes = list(graph.nodes())
            combs = combinations(nodes, self.n_nodes)
            neighs = [self._remove_node(graph, node_ids) for node_ids in combs]
        else:
            neighs = [self._remove(graph, self.n_nodes)
                      for i in range(self.n_neighbors)]
        return neighs

    def _remove_node(self, gg, node_ids):
        g = gg.copy()
        for node_id in node_ids:
            if node_id in set(g.nodes()) and len(g) > 1:
                g.remove_node(node_id)
                max_cc = max(nx.connected_components(g), key=lambda x: len(x))
                g = nx.subgraph(g, max_cc).copy()
        return g

    def _remove(self, gg, n_nodes):
        g = gg.copy()
        node_ids = [node_id for node_id in g.nodes()]
        random.shuffle(node_ids)
        node_ids = node_ids[:n_nodes]
        g = self._remove_node(g, node_ids)
        return g
