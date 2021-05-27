#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import random
import logging
logger = logging.getLogger(__name__)


class NeighborhoodNodeLabelMutation(object):

    def __init__(self, n_nodes=1, n_neighbors=10, part_importance_estimator=None):
        self.part_importance_estimator = part_importance_estimator
        self.n_neighbors = n_neighbors
        self.n_nodes = n_nodes

    def set(self, labels, probabilities=None):
        if probabilities is None:
            probabilities = [1.0 / len(labels)] * len(labels)
        self.labels = labels
        self.probabilities = probabilities

    def fit(self, graphs, targets=None):
        """fit."""
        labels = set([g.nodes[u]['label'] for g in graphs for u in g.nodes()])
        id2label_map = dict(enumerate(labels))
        label2id_map = dict((v, k) for k, v in id2label_map.items())

        n_cols = max(id2label_map) + 1
        n_rows = len(graphs)
        m = np.zeros((n_rows, n_cols))
        for i, g in enumerate(graphs):
            for u in g.nodes():
                j = label2id_map[g.nodes[u]['label']]
                m[i, j] += 1
        self.probabilities = np.mean((m.T / np.sum(m, axis=1)).T, axis=0)
        self.labels = [id2label_map[k] for k in sorted(id2label_map)]

        return self

    def _relabel(self, graph, n_mutations, labels, probabilities):
        g = graph.copy()
        new_labels = np.random.choice(
            labels, size=n_mutations, replace=True, p=probabilities)

        node_ids = list(g.nodes())
        random.shuffle(node_ids)
        node_ids = node_ids[:n_mutations]
        for l, u in zip(new_labels, node_ids):
            g.nodes[u]['label'] = l
        return g

    def neighbors(self, graph):
        """neighbors."""
        if self.n_neighbors is None:
            return self.all_neighbors(graph)
        else:
            return self.sample_neighbors(graph)

    def sample_neighbors(self, graph):
        """sample_neighbors."""
        neighs = [self._relabel(graph, self.n_nodes, self.labels, self.probabilities)
                  for i in range(self.n_neighbors)]
        return neighs

    def all_neighbors(self, graph):
        """all_neighbors."""
        graphs = []
        node_ids = list(graph.nodes())
        for node_id in node_ids:
            for label in self.labels:
                g = graph.copy()
                g.nodes[node_id]['label'] = label
                graphs.append(g)
        return graphs
