#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import networkx as nx
import random
import logging
logger = logging.getLogger(__name__)


class NeighborhoodNodeSmooth(object):

    def __init__(self, n_nodes=1, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.n_nodes = n_nodes

    def fit(self, graphs, targets):
        """fit."""
        return self

    def _smooth(self, graph, n_nodes):
        g = graph.copy()
        node_ids = list(g.nodes())
        random.shuffle(node_ids)
        counter = 0
        for u in node_ids:
            neigh_list = list(g.neighbors(u))
            if len(neigh_list) >= 2:
                v, k = np.random.choice(neigh_list, 2, replace=False)
                ev = g.edges[u, v]['label']
                ek = g.edges[u, k]['label']
                el = np.random.choice([ev, ek], 1)[0]
                g.remove_node(u)
                g.add_edge(v, k, label=el)
                counter += 1
                if counter >= n_nodes:
                    break
        max_cc = max(nx.connected_components(g), key=lambda x: len(x))
        gg = nx.subgraph(g, max_cc).copy()
        return gg

    def neighbors(self, graph):
        """neighbors."""
        neighs = [self._smooth(graph, self.n_nodes)
                  for i in range(self.n_neighbors)]
        return neighs
