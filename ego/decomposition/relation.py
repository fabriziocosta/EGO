#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import itertools
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


@curry
def decompose_relation(relation_graph_component, graph_component, type_of='single'):
    new_subgraphs_list = []
    new_signatures_list = []

    for g, signature_C in zip(relation_graph_component.subgraphs, relation_graph_component.signatures):
        g_node_set = set(g.nodes())
        subcomponents = []
        new_signatures = []
        for m, signature in zip(graph_component.subgraphs, graph_component.signatures):
            m_node_set = set(m.nodes())
            # check the node set intersection
            intr_size = len(g_node_set.intersection(m_node_set))
            if type_of == 'single':
                condition = (intr_size == 1)
            elif type_of == 'partial':
                condition = intr_size >= 1
            elif type_of == 'total':
                condition = intr_size == len(g_node_set)
            else:
                condition = False
            if condition:
                # if so save the component
                subcomponents.append(m_node_set)
                new_signatures.append(signature)
        if len(subcomponents) >= 2:
            comp_combs = itertools.combinations(subcomponents, 2)
            sig_combs = itertools.combinations(new_signatures, 2)
            for component_pair, signature_pair in zip(comp_combs, sig_combs):
                component_A, component_B = component_pair
                signature_A, signature_B = signature_pair
                component = component_A.union(component_B)
                new_subgraph = get_subgraphs_from_node_components(relation_graph_component.graph, [component])
                new_subgraphs_list += new_subgraph
                new_signature = serialize(['relation', type_of], signature_C, signature_A, signature_B)
                new_signatures_list.append(new_signature)
    gc = GraphComponent(
        graph=relation_graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc
