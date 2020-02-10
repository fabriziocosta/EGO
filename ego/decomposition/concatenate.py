#!/usr/bin/env python
"""Provides scikit interface."""

from ego.component import GraphComponent


def decompose_concatenate(*graph_components):
    new_subgraphs_list = []
    new_signatures_list = []
    for graph_component in graph_components:
        for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
            new_subgraphs_list.append(subgraph)
            new_signatures_list.append(signature)
    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc
