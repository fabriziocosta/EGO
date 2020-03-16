#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import networkx as nx
import random
import logging
logger = logging.getLogger(__name__)


class NeighborhoodEdgeMove(object):

    def __init__(self, n_edges=1, n_neighbors=10):
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
        graphs = []
        node_ids = list(graph.nodes())
        node_set = set(node_ids)
        for u in node_ids:
            neigh_list = list(graph.neighbors(u))
            neigh_set = set(neigh_list)
            neigh_set.add(u)
            candidate_endpoints = list(node_set.difference(neigh_set))
            g = graph.copy()
            for v in neigh_list:
                l = g.edges[u, v]['label']
                g.remove_edge(u, v)
                for new_v in candidate_endpoints:
                    gg = g.copy()
                    gg.add_edge(u, new_v, label=l)
                    max_cc = max(nx.connected_components(gg), key=lambda x: len(x))
                    gg = nx.subgraph(gg, max_cc).copy()
                    graphs.append(gg)
        return graphs

    def sample_neighbors(self, graph):
        """sample_neighbors."""
        neighs = [self._move(graph, self.n_edges)
                  for i in range(self.n_neighbors)]
        return neighs

    def _move(self, gg, n_edges):
        g = gg.copy()
        node_ids = list(g.nodes())
        node_set = set(node_ids)
        random.shuffle(node_ids)
        node_ids = node_ids[:n_edges]
        for u in node_ids:
            neigh_list = list(g.neighbors(u))
            v = np.random.choice(neigh_list, 1)[0]
            l = g.edges[u, v]['label']
            neigh_set = set(neigh_list)
            neigh_set.add(u)
            candidate_endpoints = list(node_set.difference(neigh_set))
            new_v = np.random.choice(candidate_endpoints, 1)[0]
            g.add_edge(u, new_v, label=l)
            g.remove_edge(u, v)
        max_cc = max(nx.connected_components(g), key=lambda x: len(x))
        g = nx.subgraph(g, max_cc).copy()
        return g
