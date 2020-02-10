#!/usr/bin/env python
"""Provides scikit interface."""


import networkx as nx
from ego.component import get_subgraphs_from_graph_component
from ego.decompose import accumulate
from ego.component import GraphComponent
from toolz import curry


@curry
def node_size_decomposition(g, size=None, min_size=1, max_size=None):
    if size is not None:
        min_size = size
        max_size = size
    node_components = []
    if nx.number_of_nodes(g) >= min_size and nx.number_of_nodes(g) <= max_size:
        node_component = set(g.nodes())
        node_components.append(node_component)
    return node_components


@curry
def decompose_node_size(graph_component, size=None, min_size=1, max_size=None):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if not subgraphs:
        subgraphs = [graph_component.graph]
    node_components = accumulate(
        node_size_decomposition(
            size=size, min_size=min_size, max_size=max_size), subgraphs)
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=node_components,
        edge_components=[])
    return gc


@curry
def edge_size_decomposition(g, size=None, min_size=1, max_size=None):
    if size is not None:
        min_size = size
        max_size = size
    edge_components = []
    if nx.number_of_edges(g) >= min_size and nx.number_of_edges(g) <= max_size:
        edge_component = set(g.edges())
        edge_components.append(edge_component)
    return edge_components


@curry
def decompose_edge_size(graph_component, size=None, min_size=1, max_size=None):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if not subgraphs:
        subgraphs = [graph_component.graph]
    edge_components = accumulate(
        edge_size_decomposition(
            size=size, min_size=min_size, max_size=max_size), subgraphs)
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=[],
        edge_components=edge_components)
    return gc
