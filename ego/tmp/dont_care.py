#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
from ego.component import GraphComponent
from ego.component import get_subgraphs_from_graph_component
import itertools


def dont_care_decomposition(dont_care_subgraphs, subgraphs, min_size=1, max_size=None):
    components = []
    # for each distinct pair of subgraphs
    for g in dont_care_subgraphs:
        g_node_set = set(g.nodes())
        subcomponents = []
        for m in subgraphs:
            m_node_set = set(m.nodes())
            # check if the node set intersection has
            # size equal or larger than min_size
            intr_size = len(g_node_set.intersection(m_node_set))
            if max_size is not None:
                condition = intr_size >= min_size and intr_size <= max_size
            else:
                condition = intr_size >= min_size
            if condition:
                # if so save the component
                subcomponents.append(m_node_set)
        if len(subcomponents) >= 2:
            for component_pair in itertools.combinations(subcomponents, 2):
                component_A, component_B = component_pair
                component = component_A.union(component_B)
                components.append(component)
    return components


@curry
def decompose_dont_care(dont_care_graph_component, graph_component, min_size=1, max_size=None):
    dont_care_subgraphs = get_subgraphs_from_graph_component(dont_care_graph_component)
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if not subgraphs:
        subgraphs = [graph_component.graph]
    node_components = dont_care_decomposition(
        dont_care_subgraphs, subgraphs, min_size=min_size, max_size=max_size)
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=node_components,
        edge_components=[])
    return gc
