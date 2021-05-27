#!/usr/bin/env python
"""Provides scikit interface."""

from toolz import curry
import itertools
import networkx as nx
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


@curry
def graphlet_decomposition(graph, size=5):
    node_components = []
    for sub_nodes in itertools.combinations(graph.nodes(), size):
        subg = graph.subgraph(sub_nodes)
        if nx.is_connected(subg):
            node_components.append(tuple(sorted(set(sub_nodes))))
    return list(set(node_components))


@curry
def decompose_graphlet(graph_component, size=5):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = graphlet_decomposition(subgraph, size=size)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['graphlet', size], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def grp(*args, **kargs): 
    return decompose_graphlet(*args, **kargs)