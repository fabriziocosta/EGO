#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


@curry
def clique_and_non_clique(graph, size=2):
    cliques = nx.enumerate_all_cliques(graph)
    cliques = list(filter(lambda x: len(x) == size, cliques))
    # induce graph from cliques and return as components
    # the connected components
    nbunch = set()
    for clique in cliques:
        nbunch.update(clique)
    clique_components = cliques
    complement_nbunch = set()
    for u in graph.nodes():
        if u not in nbunch:
            complement_nbunch.add(u)
    g2 = nx.subgraph(graph, complement_nbunch)
    non_clique_components = nx.connected_components(g2)
    clique_components = [set(c) for c in clique_components]
    non_clique_components = [set(c) for c in non_clique_components]
    return clique_components, non_clique_components


@curry
def clique_decomposition(graph, size=2):
    res = clique_and_non_clique(
        graph, size=size)
    deg_components, non_deg_components = res
    return deg_components


@curry
def non_clique_decomposition(graph, size=2):
    res = clique_and_non_clique(
        graph, size=size)
    deg_components, non_deg_components = res
    return non_deg_components


@curry
def clique_non_clique_decomposition(graph, size=2):
    res = clique_and_non_clique(
        graph, size=size)
    deg_components, non_deg_components = res
    return deg_components + non_deg_components


@curry
def decompose_clique(graph_component, size=2):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = clique_decomposition(subgraph, size=size)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['clique', size], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_non_clique(graph_component, size=2):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = non_clique_decomposition(subgraph, size=size)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['non_clique', size], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_clique_and_non_clique(graph_component, size=2):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = clique_non_clique_decomposition(subgraph, size=size)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['clique_and_non_clique', size], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc
