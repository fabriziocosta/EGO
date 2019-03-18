#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
from ego.component import GraphComponent
from ego.decompose import accumulate
from ego.component import get_subgraphs_from_graph_component


@curry
def degree_and_non_degree(graph, min_size=1, max_size=None):
    deg = dict(nx.degree(graph))
    if max_size is None:
        max_size = nx.number_of_nodes(graph)
    nbunch = set([u for u in deg if max_size >= deg[u] >= min_size])
    g1 = nx.subgraph(graph, nbunch)
    deg_components = nx.connected_components(g1)
    complement_nbunch = set()
    for u in graph.nodes():
        if u not in nbunch:
            complement_nbunch.add(u)
    g2 = nx.subgraph(graph, complement_nbunch)
    non_deg_components = nx.connected_components(g2)
    return list(deg_components), list(non_deg_components)


@curry
def degree_decomposition(graph, min_size=1, max_size=None):
    res = degree_and_non_degree(
        graph, min_size=min_size, max_size=max_size)
    deg_components, non_deg_components = res
    return deg_components


@curry
def non_degree_decomposition(graph, min_size=1, max_size=None):
    res = degree_and_non_degree(
        graph, min_size=min_size, max_size=max_size)
    deg_components, non_deg_components = res
    return non_deg_components


@curry
def degree_non_degree_decomposition(graph, min_size=1, max_size=None):
    res = degree_and_non_degree(
        graph, min_size=min_size, max_size=max_size)
    deg_components, non_deg_components = res
    return deg_components + non_deg_components


@curry
def decompose(graph_component, func):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if not subgraphs:
        subgraphs = [graph_component.graph]
    node_components = accumulate(
        func,
        subgraphs)
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=node_components,
        edge_components=[])
    return gc


@curry
def decompose_degree(graph_component, min_size=2, max_size=None):
    func = degree_decomposition(min_size=min_size, max_size=max_size)
    return decompose(graph_component, func)


@curry
def decompose_non_degree(graph_component, min_size=2, max_size=None):
    func = non_degree_decomposition(min_size=min_size, max_size=max_size)
    return decompose(graph_component, func)


@curry
def decompose_degree_and_non_degree(graph_component, min_size=2, max_size=None):
    func = degree_non_degree_decomposition(min_size=min_size, max_size=max_size)
    return decompose(graph_component, func)
