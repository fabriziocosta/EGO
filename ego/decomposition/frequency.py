#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


def _frequency_decomposition(graph, node_counts, min_size=1, max_size=None, disjoint=True):
    # a component is a connected component where all nodes
    # have a count >= min_size and <= max_size
    # select nodes
    if max_size is None:
        sel_nodes = [node for node, count in node_counts.items()
                     if min_size <= count]
    else:
        if disjoint:
            sel_nodes = [node for node, count in node_counts.items()
                         if min_size <= count < max_size]
        else:
            sel_nodes = [node for node, count in node_counts.items()
                         if min_size <= count <= max_size]
    subgraphs = get_subgraphs_from_node_components(graph, [sel_nodes])
    subgraph = subgraphs[0]
    components = list(nx.connected_components(subgraph))
    node_components = [set(c) for c in components]
    new_subgraphs = get_subgraphs_from_node_components(graph, node_components)
    return new_subgraphs


@curry
def decompose_frequency(graph_component, min_size=1, max_size=None, disjoint=True):
    """decompose_frequency."""
    new_subgraphs_list = []
    new_signatures_list = []

    signature = '*'.join(sorted(set(graph_component.signatures)))
    # count the number of components for each node
    node_counts = dict()
    for subgraph in graph_component.subgraphs:
        for node in subgraph.nodes():
            if node in node_counts:
                node_counts[node] += 1
            else:
                node_counts[node] = 1
    # a component is a connected component where all nodes
    # have a count >= min_size and <= max_size
    new_subgraphs = _frequency_decomposition(
        graph_component.graph, node_counts, min_size, max_size, disjoint)
    new_signature = serialize(
        ['frequency', min_size, max_size], signature)
    new_signatures = [new_signature] * len(new_subgraphs)
    new_subgraphs_list += new_subgraphs
    new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_partition_frequency(graph_component, step_size=1, num_intervals=None, disjoint=True):
    """decompose_partition_frequency."""
    new_subgraphs_list = []
    new_signatures_list = []

    signature = '*'.join(sorted(set(graph_component.signatures)))
    # count the number of components for each node
    node_counts = dict()
    for subgraph in graph_component.subgraphs:
        for node in subgraph.nodes():
            if node in node_counts:
                node_counts[node] += 1
            else:
                node_counts[node] = 1
    # compute range of count values
    count_values = [value for value in node_counts.values()]
    min_count_value, max_count_value = min(count_values), max(count_values)
    if disjoint:
        max_count_value = max_count_value + 1
    if num_intervals is not None:
        step_size = (max_count_value - min_count_value) // num_intervals
    # a component is a connected component where all nodes
    # have a count included within step_size
    for min_size in range(min_count_value, max_count_value, step_size):
        max_size = min_size + step_size
        new_subgraphs = _frequency_decomposition(
            graph_component.graph, node_counts, min_size, max_size, disjoint)
        new_signature = serialize(
            ['frequency', min_size, max_size], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures
    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def frq(*args, **kargs): 
    return decompose_frequency(*args, **kargs)

def prtfrq(*args, **kargs): 
    return decompose_partition_frequency(*args, **kargs)
