#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import random
import logging
logger = logging.getLogger(__name__)


class NeighborhoodEdgeAdd(object):

    def __init__(self, n_edges=1, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.n_edges = n_edges

    def set(self, labels, probabilities=None):
        if probabilities is None:
            probabilities = [1.0 / len(labels)] * len(labels)
        self.labels = labels
        self.probabilities = probabilities

    def fit(self, graphs, targets):
        """fit."""
        labels = set([g.edges[e]['label'] for g in graphs for e in g.edges()])
        id2label_map = dict(enumerate(labels))
        label2id_map = dict((v, k) for k, v in id2label_map.items())

        n_cols = max(id2label_map) + 1
        n_rows = len(graphs)
        m = np.zeros((n_rows, n_cols))
        for i, g in enumerate(graphs):
            for e in g.edges():
                j = label2id_map[g.edges[e]['label']]
                m[i, j] += 1
        self.probabilities = np.mean((m.T / np.sum(m, axis=1)).T, axis=0)
        self.labels = [id2label_map[k] for k in sorted(id2label_map)]

        return self

    def _add(self, graph, n_edges, labels, probabilities):
        g = graph.copy()
        new_labels = np.random.choice(
            labels, size=n_edges, replace=True, p=probabilities)

        node_ids = list(g.nodes())
        node_set = set(node_ids)
        random.shuffle(node_ids)
        node_ids = node_ids[:n_edges]
        for l, u in zip(new_labels, node_ids):
            neigh_set = set(g.neighbors(u))
            neigh_set.add(u)
            candidate_endpoints = list(node_set.difference(neigh_set))
            v = np.random.choice(candidate_endpoints, 1)[0]
            g.add_edge(u, v, label=l)
        return g

    def neighbors(self, graph):
        """neighbors."""
        neighs = [self._add(graph, self.n_edges, self.labels, self.probabilities)
                  for i in range(self.n_neighbors)]
        return neighs
