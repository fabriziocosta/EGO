#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


def connected_components(graph, nbunch):
    g = nx.subgraph(graph, nbunch)
    ccomponents = nx.connected_components(g)
    return ccomponents


@curry
def central_and_non_central(graph, k_top=2):
    n_dict = nx.betweenness_centrality(graph)
    central_ids = sorted(n_dict, key=lambda x: n_dict[x], reverse=True)[:k_top]
    central_components = connected_components(graph, set(central_ids))
    non_central_ids = sorted(n_dict, key=lambda x: n_dict[x], reverse=False)[:k_top]
    non_central_components = connected_components(graph, set(non_central_ids))
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
def decompose_central(graph_component, k_top=2):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = central_decomposition(subgraph, k_top)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['central', k_top], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_non_central(graph_component, k_top=2):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = non_central_decomposition(subgraph, k_top)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['non_central', k_top], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_central_and_non_central(graph_component, k_top=2):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = central_and_non_central_decomposition(subgraph, k_top)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['central_and_non_central', k_top], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc

# TODO: mark centrality of each node as the rank integer
# rank from center or from periphery
