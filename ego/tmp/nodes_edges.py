#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
from ego.component import GraphComponent
from ego.decompose import accumulate
from ego.component import get_subgraphs_from_graph_component


def nodes_and_edges(graph):
    nodes_components = [set([u]) for u in graph.nodes()]
    edges_components = [set([u, v]) for u, v in graph.edges()]
    return nodes_components, edges_components


def nodes_decomposition(graph):
    res = nodes_and_edges(graph)
    node_components, edge_components = res
    return node_components


def edges_decomposition(graph):
    res = nodes_and_edges(graph)
    node_components, edge_components = res
    return edge_components


def nodes_edges_decomposition(graph):
    res = nodes_and_edges(graph)
    node_components, edge_components = res
    return node_components + edge_components


@curry
def decompose(graph_component, func):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if len(subgraphs) == 0:
        subgraphs = [graph_component.graph]
    node_components = accumulate(
        func,
        subgraphs)
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=node_components,
        edge_components=[])
    return gc


def decompose_nodes(graph_component):
    func = nodes_decomposition
    return decompose(graph_component, func)


def decompose_edges(graph_component):
    func = edges_decomposition
    return decompose(graph_component, func)


def decompose_nodes_and_edges(graph_component):
    func = nodes_edges_decomposition
    return decompose(graph_component, func)
