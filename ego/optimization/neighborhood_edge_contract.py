#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import random
import logging
logger = logging.getLogger(__name__)


class NeighborhoodEdgeContract(object):

    def __init__(self, n_edges=1, n_neighbors=10, part_importance_estimator=None):
        self.part_importance_estimator = part_importance_estimator
        self.n_neighbors = n_neighbors
        self.n_edges = n_edges

    def fit(self, graphs, targets):
        """fit."""
        return self

    def _select_edge_endpoints(self, edge):
        v = np.random.choice(edge, 1)[0]
        # call the other k
        if v == edge[0]:
            k = edge[1]
        else:
            k = edge[0]
        return v, k

    def _contract(self, graph, n_edges):
        g = graph.copy()
        for i in range(n_edges):
            # select an edge at random
            edges = list(g.edges())
            random.shuffle(edges)
            edge = edges[0]
            # select one endpoint at random, call it v and the other k
            v, k = self._select_edge_endpoints(edge)
            v_neighbors = set(g.neighbors(v))
            v_neighbors.remove(k)
            k_neighbors = set(g.neighbors(k))
            k_neighbors.remove(v)
            # create a new node u: this will be the merged node
            u = max(g.nodes()) + 1
            # give it the label of v (one of the two endpoints at random)
            g.add_node(u, label=g.nodes[v]['label'])
            # rewire all the neighbors of v and k to u
            for v_neighbor in v_neighbors:
                g.add_edge(u, v_neighbor, label=g.edges[v, v_neighbor]['label'])
            for k_neighbor in k_neighbors:
                g.add_edge(u, k_neighbor, label=g.edges[k, k_neighbor]['label'])
            # remove the two nodes v,k
            if v in list(g.nodes()):
                g.remove_node(v)
            if k in list(g.nodes()):
                g.remove_node(k)
        return g

    def neighbors(self, graph):
        """neighbors."""
        neighs = [self._contract(graph, self.n_edges) for i in range(self.n_neighbors)]
        return neighs
