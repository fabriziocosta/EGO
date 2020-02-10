#!/usr/bin/env python
"""Provides scikit interface."""


from ego.component import GraphComponent, serialize
import networkx as nx
from toolz import curry


@curry
def decompose_abstract(graph_component, node_label=None, edge_label=None):
    graph = nx.Graph()
    for ii, (subgraph_i, signature_i) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
        i = ii + len(graph_component.graph)
        graph.add_node(i + 1, label=signature_i, isa=list(subgraph_i.nodes()))
        for jj, (subgraph_j, signature_j) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
            if jj > ii:
                j = jj + len(graph_component.graph)
                intersect = set(subgraph_i.nodes()) & set(subgraph_j.nodes())
                if intersect:     # Not empty
                    graph.add_edge(i + 1, j + 1, label=len(intersect), isa=list(intersect))

    if node_label:
        nx.set_node_attributes(graph, node_label, 'label')
    if edge_label:
        nx.set_edge_attributes(graph, edge_label, 'label')
    signature = '_'.join(sorted(set(graph_component.signatures)))
    new_signature = serialize(['abstract'], signature)

    gc = GraphComponent(
        graph=graph,
        subgraphs=[graph],
        signatures=[new_signature])
    return gc


@curry
def decompose_abstract_and_non_abstract(graph_component, node_label=None, edge_label=None, isa_label='isa'):
    graph = nx.Graph(graph_component.graph)
    for ii, (subgraph_i, signature_i) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
        i = ii + len(graph_component.graph)
        graph.add_node(i + 1, label=signature_i, isa=list(subgraph_i.nodes()))
        for v in subgraph_i.nodes():
            graph.add_edge(i + 1, v, label=isa_label, nesting=True)
        for jj, (subgraph_j, signature_j) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
            if jj > ii:
                j = jj + len(graph_component.graph)
                intersect = set(subgraph_i.nodes()) & set(subgraph_j.nodes())
                if intersect:     # Not empty
                    graph.add_edge(i + 1, j + 1, label=len(intersect), isa=list(intersect))

    if node_label:
        nx.set_node_attributes(graph, node_label, 'label')
    if edge_label:
        nx.set_edge_attributes(graph, edge_label, 'label')
    signature = '_'.join(sorted(set(graph_component.signatures)))
    new_signature = serialize(['abstract_non_abstract'], signature)

    gc = GraphComponent(
        graph=graph,
        subgraphs=[graph],
        signatures=[new_signature])
    return gc
