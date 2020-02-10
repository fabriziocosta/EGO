#!/usr/bin/env python
"""Provides scikit interface."""


from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components
from networkx.algorithms.community import greedy_modularity_communities


def decompose_communities(graph_component):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = greedy_modularity_communities(subgraph)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['communities'], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc
