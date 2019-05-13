#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
from ego.component import GraphComponent
from ego.decompose import accumulate
from ego.component import get_subgraphs_from_graph_component


@curry
def positive_and_negative(graph, threshold=0, part_importance_estimator=None):
    codes, fragments = part_importance_estimator.encoding_func(graph)
    positive_components = [set(fragment.nodes())
                           for fragment, code in zip(fragments, codes)
                           if part_importance_estimator.importance_dict.get(
        code, 0) > threshold]

    negative_components = [set(fragment.nodes())
                           for fragment, code in zip(fragments, codes)
                           if part_importance_estimator.importance_dict.get(
        code, 0) <= threshold]

    return list(positive_components), list(negative_components)


@curry
def positive_decomposition(graph, threshold=0, part_importance_estimator=None):
    res = positive_and_negative(
        graph,
        threshold=threshold,
        part_importance_estimator=part_importance_estimator)
    positive_components, negative_components = res
    return positive_components


@curry
def negative_decomposition(graph, threshold=0, part_importance_estimator=None):
    res = positive_and_negative(
        graph,
        threshold=threshold,
        part_importance_estimator=part_importance_estimator)
    positive_components, negative_components = res
    return negative_components


@curry
def positive_and_negative_decomposition(graph, threshold=0, part_importance_estimator=None):
    res = positive_and_negative(
        graph,
        threshold=threshold,
        part_importance_estimator=part_importance_estimator)
    positive_components, negative_components = res
    return positive_components + negative_components


@curry
def decompose(graph_component, func):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if not subgraphs:
        subgraphs = [graph_component.graph]
    node_components = accumulate(
        func,
        subgraphs)
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=node_components,
        edge_components=[])
    return gc


@curry
def decompose_positive(graph_component, threshold=0, part_importance_estimator=None):
    func = positive_decomposition(
        threshold=threshold,
        part_importance_estimator=part_importance_estimator)
    return decompose(graph_component, func)


@curry
def decompose_negative(graph_component, threshold=0, part_importance_estimator=None):
    func = negative_decomposition(
        threshold=threshold,
        part_importance_estimator=part_importance_estimator)
    return decompose(graph_component, func)


@curry
def decompose_positive_and_negative(graph_component, threshold=0, part_importance_estimator=None):
    func = positive_and_negative_decomposition(
        threshold=threshold,
        part_importance_estimator=part_importance_estimator)
    return decompose(graph_component, func)
