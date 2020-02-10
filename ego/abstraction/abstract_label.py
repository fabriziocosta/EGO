#!/usr/bin/env python
"""Provides scikit interface."""


import networkx as nx
from toolz import curry


@curry
def preprocess_abstract_label(graph, node_label=None, edge_label=None):
    g = nx.Graph(graph)
    if node_label is not None:
        for u in g.nodes:
            g.nodes[u]['label'] = node_label
    if edge_label is not None:
        for u, v in g.edges():
            g.edges[u, v]['label'] = edge_label
    return g
