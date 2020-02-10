#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
import numpy as np
from ego.component import GraphComponent, serialize, get_subgraphs_from_graph_component
from ego.decompose import accumulate


@curry
def discriminative_and_non_discriminative(graph, min_size=2, max_size=8, node_scoring_func=None, threshold=0):
    n_dict = node_scoring_func(graph)
    threshold = np.percentile(n_dict.values(), threshold * 100)
    ids = [node_id for node_id in n_dict if n_dict[node_id] > threshold]
    nbunch = set(ids)
    g1 = nx.subgraph(graph, nbunch)
    discriminative_components = nx.connected_components(g1)
    discriminative_components = [component
                                 for component in discriminative_components
                                 if min_size <= len(component) <= max_size]

    complement_nbunch = set()
    for u in graph.nodes():
        if u not in nbunch:
            complement_nbunch.add(u)
    g2 = nx.subgraph(graph, complement_nbunch)
    non_discriminative_components = nx.connected_components(g2)
    non_discriminative_components = [component
                                     for component in non_discriminative_components
                                     if min_size <= len(component) <= max_size]

    return list(discriminative_components), list(non_discriminative_components)


@curry
def discriminative_decomposition(graph, min_size=2, max_size=8, node_scoring_func=None, threshold=0):
    res = discriminative_and_non_discriminative(
        graph, min_size=min_size, max_size=max_size, node_scoring_func=node_scoring_func, threshold=threshold)
    discriminative_components, non_discriminative_components = res
    return discriminative_components


@curry
def non_discriminative_decomposition(graph, min_size=2, max_size=8, node_scoring_func=None, threshold=0):
    res = discriminative_and_non_discriminative(
        graph, min_size=min_size, max_size=max_size, node_scoring_func=node_scoring_func, threshold=threshold)
    discriminative_components, non_discriminative_components = res
    return non_discriminative_components


@curry
def discriminative_and_non_discriminative_decomposition(graph, min_size=2, max_size=8, node_scoring_func=None, threshold=0):
    res = discriminative_and_non_discriminative(
        graph, min_size=min_size, max_size=max_size, node_scoring_func=node_scoring_func, threshold=threshold)
    discriminative_components, non_discriminative_components = res
    return discriminative_components + non_discriminative_components


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
        edge_components=[],
        signature='decompose')
    return gc


@curry
def decompose_discriminative(graph_component, min_size=2, max_size=8, node_scoring_func=None, threshold=0):
    func = discriminative_decomposition(
        min_size=min_size,
        max_size=max_size,
        node_scoring_func=node_scoring_func,
        threshold=threshold)
    gc = decompose(graph_component, func)
    gc.signature = serialize(['discriminative',
                              min_size,
                              max_size,
                              node_scoring_func.__name__,
                              threshold], graph_component.signature)
    return gc


@curry
def decompose_non_discriminative(graph_component, min_size=2, max_size=8, node_scoring_func=None, threshold=0):
    func = non_discriminative_decomposition(
        min_size=min_size,
        max_size=max_size,
        node_scoring_func=node_scoring_func,
        threshold=threshold)
    gc = decompose(graph_component, func)
    gc.signature = serialize(['non_discriminative',
                              min_size,
                              max_size,
                              node_scoring_func.__name__,
                              threshold], graph_component.signature)
    return gc


@curry
def decompose_discriminative_and_non_discriminative(graph_component, min_size=2, max_size=8, node_scoring_func=None, threshold=0):
    func = discriminative_and_non_discriminative_decomposition(
        min_size=min_size,
        max_size=max_size,
        node_scoring_func=node_scoring_func,
        threshold=threshold)
    gc = decompose(graph_component, func)
    gc.signature = serialize(['discriminative_and_non_discriminative',
                              min_size,
                              max_size,
                              node_scoring_func.__name__,
                              threshold],
                             graph_component.signature)
    return gc
