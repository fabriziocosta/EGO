#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import random
import logging
logger = logging.getLogger(__name__)


class NeighborhoodNodeAdd(object):

    def __init__(self, n_nodes=1, n_neighbors=10, part_importance_estimator=None):
        self.part_importance_estimator = part_importance_estimator
        self.n_neighbors = n_neighbors
        self.n_nodes = n_nodes

    def set_node_labels(self, labels, probabilities=None):
        """set_node_labels."""
        if probabilities is None:
            probabilities = [1.0 / len(labels)] * len(labels)
        self.node_labels = labels
        self.node_label_probabilities = probabilities

    def set_edge_labels(self, labels, probabilities=None):
        """set_edge_labels."""
        if probabilities is None:
            probabilities = [1.0 / len(labels)] * len(labels)
        self.edge_labels = labels
        self.edge_label_probabilities = probabilities

    def _fit_node_labels(self, graphs, targets):
        labels = set([g.nodes[u]['label'] for g in graphs for u in g.nodes()])
        assert len(labels) > 0, 'Error: no node labels are present'
        id2label_map = dict(enumerate(labels))
        label2id_map = dict((v, k) for k, v in id2label_map.items())

        n_cols = max(id2label_map) + 1
        n_rows = len(graphs)
        m = np.zeros((n_rows, n_cols))
        for i, g in enumerate(graphs):
            for u in g.nodes():
                j = label2id_map[g.nodes[u]['label']]
                m[i, j] += 1
        self.node_label_probabilities = np.mean((m.T / np.sum(m, axis=1)).T, axis=0)
        self.node_labels = [id2label_map[k] for k in sorted(id2label_map)]

    def _fit_edge_labels(self, graphs, targets):
        labels = set([g.edges[e]['label'] for g in graphs for e in g.edges()])
        assert len(labels) > 0, 'Error: no edge labels are present'
        id2label_map = dict(enumerate(labels))
        label2id_map = dict((v, k) for k, v in id2label_map.items())

        n_cols = max(id2label_map) + 1
        n_rows = len(graphs)
        m = np.zeros((n_rows, n_cols))
        for i, g in enumerate(graphs):
            for e in g.edges():
                j = label2id_map[g.edges[e]['label']]
                m[i, j] += 1
        self.edge_label_probabilities = np.mean((m.T / np.sum(m, axis=1)).T, axis=0)
        self.edge_labels = [id2label_map[k] for k in sorted(id2label_map)]

    def fit(self, graphs, targets):
        """fit."""
        self._fit_node_labels(graphs, targets)
        self._fit_edge_labels(graphs, targets)
        return self

    def _add(self, graph, n_nodes, node_labels, node_label_probabilities, edge_labels, edge_label_probabilities):
        g = graph.copy()
        if len(node_label_probabilities) == 1:
            new_node_labels = [node_labels[0]]*n_nodes
        else:
            new_node_labels = np.random.choice(
                node_labels, size=n_nodes, replace=True, p=node_label_probabilities)
        if len(node_label_probabilities) == 1:
            new_edge_labels = [edge_labels[0]]*n_nodes
        else:
            new_edge_labels = np.random.choice(
                edge_labels, size=n_nodes, replace=True, p=edge_label_probabilities)

        node_ids = list(g.nodes())
        for nl, el in zip(new_node_labels, new_edge_labels):
            u = max(g.nodes()) + 1
            g.add_node(u, label=nl)
            v = np.random.choice(node_ids, 1)[0]
            g.add_edge(u, v, label=el)
        return g

    def neighbors(self, graph):
        """neighbors."""
        neighs = [self._add(graph, self.n_nodes, self.node_labels, self.node_label_probabilities, self.edge_labels, self.edge_label_probabilities)
                  for i in range(self.n_neighbors)]
        return neighs
