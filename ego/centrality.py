#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
from ego.component import GraphComponent
from ego.decompose import accumulate
from ego.component import get_subgraphs_from_graph_component


@curry
def central_and_non_central(graph, k_top=2):
    n_dict = nx.betweenness_centrality(graph)
    ids = sorted(n_dict, key=lambda x: n_dict[x], reverse=True)[:k_top]
    nbunch = set(ids)
    g1 = nx.subgraph(graph, nbunch)
    central_components = nx.connected_components(g1)
    complement_nbunch = set()
    for u in graph.nodes():
        if u not in nbunch:
            complement_nbunch.add(u)
    g2 = nx.subgraph(graph, complement_nbunch)
    non_central_components = nx.connected_components(g2)
    return list(central_components), list(non_central_components)


@curry
def central_decomposition(graph, k_top=2):
    res = central_and_non_central(graph, k_top=k_top)
    central_components, non_central_components = res
    return central_components


@curry
def non_central_decomposition(graph, k_top=2):
    res = central_and_non_central(graph, k_top=k_top)
    central_components, non_central_components = res
    return non_central_components


@curry
def central_and_non_central_decomposition(graph, k_top=2):
    res = central_and_non_central(graph, k_top=k_top)
    central_components, non_central_components = res
    return central_components + non_central_components


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
def decompose_central(graph_component, k_top=2):
    func = central_decomposition(k_top=k_top)
    return decompose(graph_component, func)


@curry
def decompose_non_central(graph_component, k_top=2):
    func = non_central_decomposition(k_top=k_top)
    return decompose(graph_component, func)


@curry
def decompose_central_and_non_central(graph_component, k_top=2):
    func = central_and_non_central_decomposition(k_top=k_top)
    return decompose(graph_component, func)
