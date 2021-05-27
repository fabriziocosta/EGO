#!/usr/bin/env python
"""Provides scikit interface."""

from ego.component import GraphComponent
from toolz import curry

def decompose_nodes_relabel_degree(graph_component):
    assert(len(graph_component.subgraphs[0]) == len(graph_component.graph)), 'Node relabeling is allowed only on the original graph, not on components'
    graph = graph_component.graph.copy()
    for u in graph.nodes():
        graph.nodes[u]['label'] = graph.degree[u]
    for u, v in graph.edges():
        graph.edges[u, v]['label'] = '::'

    signatures = ['node_relabel_degree']
    subgraphs = [graph]

    gc = GraphComponent(
        graph=graph,
        subgraphs=subgraphs,
        signatures=signatures)
    return gc


def decompose_nodes_relabel_null(graph_component):
    assert(len(graph_component.subgraphs[0]) == len(graph_component.graph)), 'Node relabeling is allowed only on the original graph, not on components'
    graph = graph_component.graph.copy()
    for u in graph.nodes():
        graph.nodes[u]['label'] = '::'
    for u, v in graph.edges():
        graph.edges[u, v]['label'] = '::'

    signatures = ['node_relabel_null']
    subgraphs = [graph]

    gc = GraphComponent(
        graph=graph,
        subgraphs=subgraphs,
        signatures=signatures)
    return gc

@curry
def decompose_nodes_relabel_mapped(graph_component, node_label_map=dict(), edge_label_map=dict()):
    assert(len(graph_component.subgraphs[0]) == len(graph_component.graph)), 'Node relabeling is allowed only on the original graph, not on components'
    graph = graph_component.graph.copy()
    for u in graph.nodes():
        graph.nodes[u]['label'] = node_label_map.get(graph.nodes[u]['label'], '::')
    for u, v in graph.edges():
        graph.edges[u, v]['label'] = edge_label_map.get(graph.edges[u, v]['label'], '::')

    signatures = ['node_relabel_null']
    subgraphs = [graph]

    gc = GraphComponent(
        graph=graph,
        subgraphs=subgraphs,
        signatures=signatures)
    return gc


def ndsrlbdgr(*args, **kargs): 
    return decompose_nodes_relabel_degree(*args, **kargs)


def ndsrlbnll(*args, **kargs): 
    return decompose_nodes_relabel_null(*args, **kargs)

def ndsrlbmpd(*args, **kargs): 
    return decompose_nodes_relabel_mapped(*args, **kargs)
