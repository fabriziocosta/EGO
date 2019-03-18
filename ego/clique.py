#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
from ego.component import GraphComponent
from ego.decompose import accumulate
from ego.component import get_subgraphs_from_graph_component


@curry
def clique_and_non_clique(graph, min_size=1, max_size=None):
    cliques = nx.enumerate_all_cliques(graph)
    if max_size is None:
        max_size = nx.number_of_nodes(graph)
    cliques = filter(lambda x: max_size >= len(x) >= min_size, cliques)
    # induce graph from cliques and return as components
    # the connected components
    nbunch = set()
    for clique in cliques:
        nbunch.update(clique)
    g1 = nx.subgraph(graph, nbunch)
    clique_components = nx.connected_components(g1)
    complement_nbunch = set()
    for u in graph.nodes():
        if u not in nbunch:
            complement_nbunch.add(u)
    g2 = nx.subgraph(graph, complement_nbunch)
    non_clique_components = nx.connected_components(g2)
    return list(clique_components), list(non_clique_components)


@curry
def clique_decomposition(graph, min_size=1, max_size=None):
    res = clique_and_non_clique(
        graph, min_size=min_size, max_size=max_size)
    deg_components, non_deg_components = res
    return deg_components


@curry
def non_clique_decomposition(graph, min_size=1, max_size=None):
    res = clique_and_non_clique(
        graph, min_size=min_size, max_size=max_size)
    deg_components, non_deg_components = res
    return non_deg_components


@curry
def clique_non_clique_decomposition(graph, min_size=1, max_size=None):
    res = clique_and_non_clique(
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
def decompose_clique(graph_component, min_size=2, max_size=None):
    func = clique_decomposition(min_size=min_size, max_size=max_size)
    return decompose(graph_component, func)


@curry
def decompose_non_clique(graph_component, min_size=2, max_size=None):
    func = non_clique_decomposition(min_size=min_size, max_size=max_size)
    return decompose(graph_component, func)


@curry
def decompose_clique_and_non_clique(graph_component, min_size=2, max_size=None):
    func = clique_non_clique_decomposition(min_size=min_size, max_size=max_size)
    return decompose(graph_component, func)
