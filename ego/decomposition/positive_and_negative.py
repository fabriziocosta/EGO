#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import numpy as np
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


@curry
def positive_and_negative(graph, ktop=0, part_importance_estimator=None):
    codes, fragments = part_importance_estimator.encoding_func(graph)
    scores = [part_importance_estimator.importance_dict.get(code, 0) for code in codes]
    if ktop > len(scores) - 1:
        ktop = len(scores) - 1
    ids = np.argsort(scores)
    positive_components = [set(fragments[id].nodes()) for id in ids[-ktop:]]
    negative_components = [set(fragments[id].nodes()) for id in ids[:ktop]]
    return list(positive_components), list(negative_components)


@curry
def positive_decomposition(graph, ktop=0, part_importance_estimator=None):
    res = positive_and_negative(
        graph,
        ktop=ktop,
        part_importance_estimator=part_importance_estimator)
    positive_components, negative_components = res
    return positive_components


@curry
def negative_decomposition(graph, ktop=0, part_importance_estimator=None):
    res = positive_and_negative(
        graph,
        ktop=ktop,
        part_importance_estimator=part_importance_estimator)
    positive_components, negative_components = res
    return negative_components


@curry
def positive_and_negative_decomposition(graph, ktop=0, part_importance_estimator=None):
    res = positive_and_negative(
        graph,
        ktop=ktop,
        part_importance_estimator=part_importance_estimator)
    positive_components, negative_components = res
    return positive_components + negative_components


@curry
def decompose_positive(graph_component, ktop=0, part_importance_estimator=None):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = positive_decomposition(
            subgraph, ktop=ktop, part_importance_estimator=part_importance_estimator)
        new_subgraphs = get_subgraphs_from_node_components(
            graph_component.graph, components)
        new_signature = serialize(['positive',
                                   ktop], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_negative(graph_component, ktop=0, part_importance_estimator=None):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = negative_decomposition(
            subgraph, ktop=ktop, part_importance_estimator=part_importance_estimator)
        new_subgraphs = get_subgraphs_from_node_components(
            graph_component.graph, components)
        new_signature = serialize(['negative',
                                   ktop], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_positive_and_negative(graph_component, ktop=0, part_importance_estimator=None):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = positive_and_negative_decomposition(
            subgraph, ktop=ktop, part_importance_estimator=part_importance_estimator)
        new_subgraphs = get_subgraphs_from_node_components(
            graph_component.graph, components)
        new_signature = serialize(['positive_and_negative',
                                   ktop], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def pst(*args, **kargs): 
    return decompose_positive(*args, **kargs)

def ngt(*args, **kargs): 
    return decompose_negative(*args, **kargs)

def pstngt(*args, **kargs): 
    return decompose_positive_and_negative(*args, **kargs)