#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
from ego.component import GraphComponent, serialize, get_subgraphs_from_edge_components, get_subgraphs_from_node_components


@curry
def node_size_decomposition(g, min_size=1, max_size=None):
    if max_size is None:
        max_size = len(g.nodes())
    node_components = []
    if max_size >= nx.number_of_nodes(g) >= min_size:
        node_component = set(g.nodes())
        node_components.append(node_component)
    return node_components


@curry
def decompose_node_size(graph_component, min_size=1, max_size=None):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = node_size_decomposition(
            subgraph, min_size=min_size, max_size=max_size)
        new_subgraphs = get_subgraphs_from_node_components(
            graph_component.graph, components)
        new_signature = serialize(
            ['node_size', min_size, max_size], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def edge_size_decomposition(g, min_size=1, max_size=None):
    if max_size is None:
        max_size = len(g.edges())
    edge_components = []
    if max_size >= nx.number_of_edges(g) >= min_size:
        edge_component = set(g.edges())
        edge_components.append(edge_component)
    return edge_components


@curry
def decompose_edge_size(graph_component, min_size=1, max_size=None):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        edge_components = edge_size_decomposition(
            subgraph, min_size=min_size, max_size=max_size)
        new_subgraphs = get_subgraphs_from_edge_components(
            graph_component.graph, edge_components)
        new_signature = serialize(
            ['edge_size', min_size, max_size], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def ndsz(*args, **kargs): 
    return decompose_node_size(*args, **kargs)

def edgsz(*args, **kargs): 
    return decompose_edge_size(*args, **kargs)
