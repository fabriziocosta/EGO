#!/usr/bin/env python
"""Provides scikit interface."""


import networkx as nx
from ego.component import GraphComponent, serialize, get_subgraphs_from_graph_component


def decompose_connected_component(graph_component):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if len(subgraphs) == 0:
        subgraphs = [graph_component.graph]
    node_components = []
    for subgraph in subgraphs:
        components = nx.connected_components(subgraph)
        node_components += [set(c) for c in components]
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=node_components,
        edge_components=[],
        signature=serialize(['connected'], graph_component.signature))
    return gc

def cnncmp(*args, **kargs): 
    return decompose_connected_component(*args, **kargs)