#!/usr/bin/env python
"""Provides scikit interface."""


import toolz as tz
from toolz import curry
from ego.component import GraphComponent
from ego.component import get_node_components_from_abstraction_graph
from ego.component import make_abstraction_graph
from ego.component import convert, edge_to_node_components
from ego.component import transitive_reference_update


def compose(*decomposition_functions):
    return tz.compose(*decomposition_functions)


def abstract_compose(*decomposition_functions):
    def abstract_composition(in_graph_component):
        fs = decomposition_functions[::-1]
        nc = in_graph_component.node_components
        if len(nc) != 0:
            g2 = make_abstraction_graph(nc)
        else:
            out_comp = fs[0](
                in_graph_component)
            nc = out_comp.node_components + edge_to_node_components(
                out_comp.edge_components)
            g2 = make_abstraction_graph(nc)
        for f in fs[1:]:
            out_comp = f(convert(g2))
            nc = out_comp.node_components + edge_to_node_components(
                out_comp.edge_components)
            g3 = make_abstraction_graph(nc)
            g4 = transitive_reference_update(g2, g3)
            g2 = g4
        nc = get_node_components_from_abstraction_graph(g2)
        gc = GraphComponent(
            graph=in_graph_component.graph,
            node_components=nc,
            edge_components=[])
        return gc
    return abstract_composition


def head_compose(*decomposition_functions):
    """Compose the first function with all the remaining functions."""
    main_decomposition_function = decomposition_functions[0]
    argument_decomposition_functions = [f for f in decomposition_functions[1:]]

    def head_composition(in_graph_component):
        graph_components = [f(in_graph_component) for f in argument_decomposition_functions]
        return main_decomposition_function(*graph_components)
    return head_composition


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


@curry
def iterate(decompose_func, n_iter=2):
    funcs = [decompose_func] * n_iter
    return tz.compose(*funcs)
