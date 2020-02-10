#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


@curry
def decompose_node_join(graph_component, min_size=1, max_size=1):
    if max_size < min_size:
        max_size = min_size
    new_subgraphs_list = []
    new_signatures_list = []
    # for each distinct pair of subgraphs
    for i, (g, signature_i) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
        g_node_set = set(g.nodes())
        for j, (m, signature_j) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
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
                    new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, [component])
                    new_signature = serialize(['node_join', min_size, max_size], signature_i, signature_j)
                    new_subgraphs_list += new_subgraphs
                    new_signatures_list += [new_signature]

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def get_edge_set(g):
    return set([(e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
                for e in g.edges()])


@curry
def decompose_edge_join(graph_component, min_size=1, max_size=1):
    if max_size < min_size:
        max_size = min_size
    new_subgraphs_list = []
    new_signatures_list = []
    # for each distinct pair of subgraphs
    for i, (g, signature_i) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
        g_node_set = set(g.nodes())
        g_edge_set = get_edge_set(g)
        for j, (m, signature_j) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
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
                    new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, [component])
                    new_signature = serialize(['edge_join', min_size, max_size], signature_i, signature_j)
                    new_subgraphs_list += new_subgraphs
                    new_signatures_list += [new_signature]

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc
