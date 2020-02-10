#!/usr/bin/env python
"""Provides scikit interface."""


from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


def nodes_and_edges(graph):
    nodes_components = [set([u]) for u in graph.nodes()]
    edges_components = [set([u, v]) for u, v in graph.edges()]
    return nodes_components, edges_components


def nodes_decomposition(graph):
    res = nodes_and_edges(graph)
    node_components, edge_components = res
    return node_components


def edges_decomposition(graph):
    res = nodes_and_edges(graph)
    node_components, edge_components = res
    return edge_components


def nodes_edges_decomposition(graph):
    res = nodes_and_edges(graph)
    node_components, edge_components = res
    return node_components + edge_components


def decompose_nodes(graph_component):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = nodes_decomposition(subgraph)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['nodes'], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def decompose_edges(graph_component):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = edges_decomposition(subgraph)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['edges'], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def decompose_nodes_and_edges(graph_component):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = nodes_edges_decomposition(subgraph)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['nodes_and_edges'], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc
