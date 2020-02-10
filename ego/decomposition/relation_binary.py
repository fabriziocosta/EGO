#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import itertools
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


def get_intersecting_subgraphs(g_node_set, graph_component, type_of='single'):
    subcomponents = []
    new_signatures = []
    for m, s in zip(graph_component.subgraphs, graph_component.signatures):
        m_node_set = set(m.nodes())
        # check if the node set intersection has the right size
        intr_size = len(g_node_set.intersection(m_node_set))
        if type_of == 'single':
            condition = (intr_size == 1)
        elif type_of == 'partial':
            condition = intr_size >= 1
        elif type_of == 'total':
            condition = intr_size == len(g_node_set)
        else:
            condition = False
        if condition is True:
            # if so save the component
            subcomponents.append(m_node_set)
            new_signatures.append(s)
    return subcomponents, new_signatures


@curry
def decompose_relation_binary(relation_graph_component, graph_component_first, graph_component_second, type_of='single', keep_second_component=True):
    new_subgraphs_list = []
    new_signatures_list = []

    for g, signature_C in zip(relation_graph_component.subgraphs, relation_graph_component.signatures):
        g_node_set = set(g.nodes())
        subcomponents_first, signatures_first = get_intersecting_subgraphs(
            g_node_set, graph_component_first, type_of)
        subcomponents_second, signatures_second = get_intersecting_subgraphs(
            g_node_set, graph_component_second, type_of)
        if len(subcomponents_first) >= 1 and len(subcomponents_second) >= 1:
            component_pairs = itertools.product(
                subcomponents_first, subcomponents_second)
            signatures_pairs = itertools.product(signatures_first, signatures_second)
            for component_pair, signatures_pair in zip(component_pairs, signatures_pairs):
                component_A, component_B = component_pair
                signature_A, signature_B = signatures_pair
                if component_A != component_B:
                    if keep_second_component:
                        component = component_A.union(component_B)
                    else:
                        component = component_A
                    new_subgraph = get_subgraphs_from_node_components(relation_graph_component.graph, [component])
                    new_subgraphs_list += new_subgraph
                    new_signature = serialize(['relation_binary', type_of, keep_second_component], signature_C, signature_A, signature_B)
                    new_signatures_list += new_signature

    gc = GraphComponent(
        graph=relation_graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc
