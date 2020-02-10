#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


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
def decompose_degree(graph_component, min_size=2, max_size=None):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = degree_decomposition(
            subgraph, min_size=min_size, max_size=max_size)
        new_subgraphs = get_subgraphs_from_node_components(
            graph_component.graph, components)
        new_signature = serialize(
            ['degree', min_size, max_size], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_non_degree(graph_component, min_size=2, max_size=None):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = non_degree_decomposition(
            subgraph, min_size=min_size, max_size=max_size)
        new_subgraphs = get_subgraphs_from_node_components(
            graph_component.graph, components)
        new_signature = serialize(
            ['non_degree', min_size, max_size], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_degree_and_non_degree(graph_component, min_size=2, max_size=None):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = degree_non_degree_decomposition(
            subgraph, min_size=min_size, max_size=max_size)
        new_subgraphs = get_subgraphs_from_node_components(
            graph_component.graph, components)
        new_signature = serialize(
            ['degree_and_non_degree', min_size, max_size], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc
