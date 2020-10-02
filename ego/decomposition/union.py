#!/usr/bin/env python
"""Provides scikit interface."""


import networkx as nx
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


def decompose_all_union(graph_component):
    subgraphs = graph_component.subgraphs
    # Warning: as it is hard to keep track of which components are
    # finally united, we simply mangle all signatures into one
    # as the sorted union of all signatures
    new_signature = '_'.join(sorted(set(graph_component.signatures)))
    new_subgraphs = []
    new_signatures = []
    if len(subgraphs) > 0:
        g = subgraphs[0]
        for subgraph in subgraphs[1:]:
            g = nx.compose(g, subgraph)
        components = nx.connected_components(g)
        new_subgraphs = get_subgraphs_from_node_components(
            graph_component.graph, components)
        new_signature = serialize(['union'], new_signature)
        new_signatures = [new_signature] * len(new_subgraphs)

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs,
        signatures=new_signatures)
    return gc
