#!/usr/bin/env python
"""Provides scikit interface."""


import networkx as nx
from toolz import curry


def label_to_degree(graph):
    g = nx.Graph(graph)
    for u in g.nodes():
        g.nodes[u]['original_label'] = g.nodes[u]['label']
        g.nodes[u]['label'] = len(g[u])
    return g


def minor(graph):
    g = nx.Graph(graph)
    while True:
        has_contracted = False
        for u, v in g.edges():
            if g.nodes[u]['label'] == g.nodes[v]['label']:
                g = nx.contracted_edge(g, (u, v), self_loops=False)
                has_contracted = True
                break
        if has_contracted is False:
            break
    return g


@curry
def preprocess_minor_degree(graph, node_label=None, edge_label=None):
    g = label_to_degree(graph)
    g = minor(g)
    if node_label is not None:
        for u in g.nodes:
            g.nodes[u]['label'] = node_label
    if edge_label is not None:
        for u, v in g.edges():
            g.edges[u, v]['label'] = edge_label
    return g
