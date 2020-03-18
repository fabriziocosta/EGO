#!/usr/bin/env python
"""Provides scikit interface."""


import toolz as tz
from toolz import curry
from ego.component import GraphComponent
from ego.component import get_node_components_from_abstraction_graph
from ego.component import make_abstraction_graph
from ego.component import convert, edge_to_node_components
from ego.component import transitive_reference_update
from ego.decomposition.identity import decompose_identity
from ego.decomposition.concatenate import decompose_concatenate


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
        graph_components = [f(in_graph_component)
                            for f in argument_decomposition_functions]
        return main_decomposition_function(*graph_components)
    return head_composition


def args(*decomposition_functions):
    return head_compose(*decomposition_functions)


def concatenate(*decomposition_functions):
    def combined_decomposition_function(in_graph_component):
        subgraphs = []
        signatures = []
        for decomposition_function in decomposition_functions:
            graph_component = decomposition_function(in_graph_component)
            subgraphs += graph_component.subgraphs
            signatures += graph_component.signatures
        gc = GraphComponent(
            graph=in_graph_component.graph,
            subgraphs=subgraphs,
            signatures=signatures)
        return gc
    return combined_decomposition_function


def concatenate_disjunctive(*decomposition_functions):
    def combined_decomposition_function(in_graph_component):
        subgraphs = []
        signatures = []
        for decomposition_function in decomposition_functions:
            graph_component = decomposition_function(in_graph_component)
            subgraphs += graph_component.subgraphs
            signatures += graph_component.signatures
        new_signature = 'disjunction' + '*'.join(sorted(set(signatures)))
        new_signatures = [new_signature] * len(subgraphs)
        gc = GraphComponent(
            graph=in_graph_component.graph,
            subgraphs=subgraphs,
            signatures=new_signatures)
        return gc
    return combined_decomposition_function


def accumulate(func, graphs):
    return tz.reduce(lambda x, y: x + y, map(func, graphs))


@curry
def iterate(decompose_func, n_iter=2):
    funcs = [decompose_func] * n_iter
    return tz.compose(*funcs)


def do_decompose(*decomposition_functions, **kargs):
    """
    Provide the template for the composition of aggregated
    decomposition functions.

    Arguments:
        *decomposition_functions: the list of decomposition_functions
        to be aggregated.

        aggregate_function: the higher order decomposition.
        Default is concatenate. Default: decompose_concatenate

        compose_function: the decomposition applied to the result of
        the aggregtion. Default: decompose_identity.

    Returns:
        a decomposition function.

    """
    return compose(
        kargs.get('compose_function', decompose_identity),
        args(kargs.get('aggregate_function', decompose_concatenate),
             *decomposition_functions))
