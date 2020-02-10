#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
from ego.component import GraphComponent
from ego.component import get_subgraphs_from_graph_component


def node_join_decomposition(subgraphs, min_size=1, max_size=None):
    components = []
    # for each distinct pair of subgraphs
    for i, g in enumerate(subgraphs):
        g_node_set = set(g.nodes())
        for j, m in enumerate(subgraphs):
            if j > i:
                m_node_set = set(m.nodes())
                # check if the node set intersection has
                # size equal or larger than min_size
                intr_size = len(g_node_set.intersection(m_node_set))
                if max_size is not None:
                    condition = intr_size >= min_size and intr_size <= max_size
                else:
                    condition = intr_size >= min_size
                if condition:
                    # if so return the union of the nodes
                    component = g_node_set.union(m_node_set)
                    components.append(component)
    return components


@curry
def decompose_node_join(graph_component, min_size=1, max_size=None):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if not subgraphs:
        subgraphs = [graph_component.graph]
    node_components = node_join_decomposition(
        subgraphs, min_size=min_size, max_size=max_size)
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=node_components,
        edge_components=[])
    return gc


def get_edge_set(g):
    return set([(e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
                for e in g.edges()])


def edge_join_decomposition(subgraphs, min_size=1, max_size=None):
    components = []
    # for each distinct pair of subgraphs
    for i, g in enumerate(subgraphs):
        g_node_set = set(g.nodes())
        g_edge_set = get_edge_set(g)
        for j, m in enumerate(subgraphs):
            if j > i:
                m_node_set = set(m.nodes())
                m_edge_set = get_edge_set(m)
                # check if the edge set intersection has
                # size equal or larger than min_size
                intr_size = len(g_edge_set.intersection(m_edge_set))
                if max_size is not None:
                    condition = intr_size >= min_size and intr_size <= max_size
                else:
                    condition = intr_size >= min_size
                if condition:
                    # if so return the union of the nodes
                    component = g_node_set.union(m_node_set)
                    components.append(component)
    return components


@curry
def decompose_edge_join(graph_component, min_size=1, max_size=None):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if not subgraphs:
        subgraphs = [graph_component.graph]
    node_components = edge_join_decomposition(
        subgraphs, min_size=min_size, max_size=max_size)
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=node_components,
        edge_components=[])
    return gc
