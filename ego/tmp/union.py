#!/usr/bin/env python
"""Provides scikit interface."""


import networkx as nx
from ego.component import get_subgraphs_from_graph_component
from ego.component import GraphComponent


def decompose_union(graph_component):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    node_components = []
    if len(subgraphs) > 0:
        g = subgraphs[0]
        for subgraph in subgraphs[1:]:
            g = nx.compose(g, subgraph)
        components = nx.connected_components(g)
        node_components = [set(c) for c in components]
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=node_components,
        edge_components=[])
    return gc
