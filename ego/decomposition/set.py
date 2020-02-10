#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


@curry
def decompose_difference(graph_component_A, graph_component_B):
    new_subgraphs_list = []
    new_signatures_list = []

    for g, signature_A in zip(graph_component_A.subgraphs, graph_component_A.signatures):
        g_node_set = set(g.nodes())
        for m, signature_B in zip(graph_component_B.subgraphs, graph_component_B.signatures):
            m_node_set = set(m.nodes())
            # check the node set intersection
            intr_size = len(g_node_set.intersection(m_node_set))
            if intr_size >= 1 and g_node_set != m_node_set:
                component = g_node_set.difference(m_node_set)
                if len(component) > 0:
                    new_subgraph = get_subgraphs_from_node_components(
                        graph_component_A.graph, [component])
                    new_subgraphs_list += new_subgraph
                    new_signature = serialize(
                        ['difference'], signature_A, signature_B)
                    new_signatures_list.append(new_signature)
    gc = GraphComponent(
        graph=graph_component_A.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_symmetric_difference(graph_component_A, graph_component_B):
    new_subgraphs_list = []
    new_signatures_list = []

    for g, signature_A in zip(graph_component_A.subgraphs, graph_component_A.signatures):
        g_node_set = set(g.nodes())
        for m, signature_B in zip(graph_component_B.subgraphs, graph_component_B.signatures):
            m_node_set = set(m.nodes())
            # check the node set intersection
            intr_size = len(g_node_set.intersection(m_node_set))
            if intr_size >= 1 and g_node_set != m_node_set:
                component = g_node_set.symmetric_difference(m_node_set)
                if len(component) > 0:
                    new_subgraph = get_subgraphs_from_node_components(
                        graph_component_A.graph, [component])
                    new_subgraphs_list += new_subgraph
                    new_signature = serialize(
                        ['symmetric_difference'], signature_A, signature_B)
                    new_signatures_list.append(new_signature)
    gc = GraphComponent(
        graph=graph_component_A.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_union(graph_component_A, graph_component_B):
    new_subgraphs_list = []
    new_signatures_list = []

    for g, signature_A in zip(graph_component_A.subgraphs, graph_component_A.signatures):
        g_node_set = set(g.nodes())
        for m, signature_B in zip(graph_component_B.subgraphs, graph_component_B.signatures):
            m_node_set = set(m.nodes())
            # check the node set intersection
            intr_size = len(g_node_set.intersection(m_node_set))
            if intr_size >= 1 and g_node_set != m_node_set:
                component = g_node_set.union(m_node_set)
                if len(component) > 0:
                    new_subgraph = get_subgraphs_from_node_components(
                        graph_component_A.graph, [component])
                    new_subgraphs_list += new_subgraph
                    new_signature = serialize(
                        ['union'], signature_A, signature_B)
                    new_signatures_list.append(new_signature)
    gc = GraphComponent(
        graph=graph_component_A.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_intersection(graph_component_A, graph_component_B):
    new_subgraphs_list = []
    new_signatures_list = []

    for g, signature_A in zip(graph_component_A.subgraphs, graph_component_A.signatures):
        g_node_set = set(g.nodes())
        for m, signature_B in zip(graph_component_B.subgraphs, graph_component_B.signatures):
            m_node_set = set(m.nodes())
            # check the node set intersection
            intr_size = len(g_node_set.intersection(m_node_set))
            if intr_size >= 1 and g_node_set != m_node_set:
                component = g_node_set.intersection(m_node_set)
                if len(component) > 0:
                    new_subgraph = get_subgraphs_from_node_components(
                        graph_component_A.graph, [component])
                    new_subgraphs_list += new_subgraph
                    new_signature = serialize(
                        ['intersection'], signature_A, signature_B)
                    new_signatures_list.append(new_signature)
    gc = GraphComponent(
        graph=graph_component_A.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc
