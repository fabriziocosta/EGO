#!/usr/bin/env python
"""Provides scikit interface."""


import networkx as nx
from ego.component import get_subgraphs_from_graph_component
from ego.decompose import accumulate
from ego.component import GraphComponent
from toolz import curry


@curry
def path_decomposition(g, length=None, min_len=1, max_len=None):
    if length is not None:
        min_len = length
        max_len = length
    edge_components = []
    for u in g.nodes():
        for v in g.nodes():
            if v > u:
                for path in nx.all_shortest_paths(
                        g, source=u, target=v):
                    edge_component = set()
                    if len(path) >= min_len + 1 and len(path) <= max_len + 1:
                        for i, u in enumerate(path[:-1]):
                            w = path[i + 1]
                            if u < w:
                                edge_component.add((u, w))
                            else:
                                edge_component.add((w, u))
                    if edge_component:
                        edge_components.append(edge_component)
    return edge_components


@curry
def decompose_path(graph_component, length=None, min_len=1, max_len=None):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if not subgraphs:
        subgraphs = [graph_component.graph]
    edge_components = accumulate(
        path_decomposition(length=length, min_len=min_len, max_len=max_len),
        subgraphs)
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=[],
        edge_components=edge_components)
    return gc
