#!/usr/bin/env python
"""Provides scikit interface."""

import networkx as nx
from ego.component import GraphComponent
from ego.component import get_subgraphs_from_graph_component
from ego.decompose import accumulate
from toolz import curry
import itertools


@curry
def graphlet_decomposition(graph, size=5):
    node_components = []
    for sub_nodes in itertools.combinations(graph.nodes(), size):
        subg = graph.subgraph(sub_nodes)
        if nx.is_connected(subg):
            node_components.append(set(sub_nodes))
    return node_components


@curry
def decompose_graphlet(graph_component, size=5):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if not subgraphs:
        subgraphs = [graph_component.graph]
    nc = accumulate(
        graphlet_decomposition(size=size),
        subgraphs)
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=nc,
        edge_components=[])
    return gc
