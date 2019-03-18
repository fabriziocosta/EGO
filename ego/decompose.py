#!/usr/bin/env python
"""Provides scikit interface."""


import toolz as tz
from ego.component import GraphComponent


def compose(*decomposition_functions):
    return tz.compose(*decomposition_functions)


def concatenate(*decomposition_functions):
    def combined_decomposition_function(in_graph_component):
        all_nc = []
        all_ec = []
        for decomposition_function in decomposition_functions:
            graph_component = decomposition_function(in_graph_component)
            g = graph_component.graph
            nc = graph_component.node_components
            ec = graph_component.edge_components
            all_nc += nc
            all_ec += ec
        g = in_graph_component.graph
        gc = GraphComponent(
            graph=g, node_components=all_nc, edge_components=all_ec)
        return gc
    return combined_decomposition_function


def accumulate(func, graphs):
    return tz.reduce(lambda x, y: x + y, map(func, graphs))
