#!/usr/bin/env python
"""Provides scikit interface."""


from ego.component import GraphComponent, serialize
import networkx as nx
from toolz import curry
from ego.vectorize import set_feature_size
from ego.encode import make_encoder
from ego.decompose import do_decompose


def make_abstract_graph(graphs, decomposition=None, preprocessors=None):
    df = do_decompose(decomposition, compose_function=decompose_abstract_and_non_abstract)
    #df = do_decompose(decomposition, compose_function=decompose_abstract)

    feature_size, bitmask = set_feature_size(nbits=14)
    encoding_func = make_encoder(df, preprocessors=preprocessors, bitmask=bitmask, seed=1)

    abstract_graphs = []
    for g in graphs:
        codes, fragments = encoding_func(g)
        assert(len(fragments) == 1), "expecting 1 fragment but got:%d" % len(fragments)
        abstract_graph = fragments[0]
        abstract_graphs.append(abstract_graph)
    return abstract_graphs


def _get_edges(graph, nbunch):
    edges = []
    for u in nbunch:
        neighs = graph.neighbors(u)
        for v in neighs:
            if u < v:
                edges.append((u, v))
            else:
                edges.append((v, u))
    return edges


@curry
def decompose_abstract(graph_component, node_label=None, edge_label=None, mode='intersection'):
    graph = nx.Graph()
    n = len(graph_component.graph)
    for ii, (subgraph_i, signature_i) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
        i = ii + n
        graph.add_node(i, label=signature_i, isa=list(subgraph_i.nodes()))
        for jj, (subgraph_j, signature_j) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
            if jj > ii:
                j = jj + n
                if mode == 'intersection':
                    intersect = set(subgraph_i.nodes()) & set(subgraph_j.nodes())
                    if intersect:     # Not empty
                        graph.add_edge(i, j, label=len(intersect), isa=list(intersect))
                if mode == 'edge':
                    subgraph_i_all_edges = _get_edges(graph_component.graph, subgraph_i.nodes())
                    subgraph_j_all_edges = _get_edges(graph_component.graph, subgraph_j.nodes())
                    intersect = set(subgraph_i_all_edges) & set(subgraph_j_all_edges)
                    if intersect:     # Not empty
                        graph.add_edge(i, j, label=len(intersect), isa=list(intersect))
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
def decompose_abstract_and_non_abstract(graph_component, node_label=None, edge_label=None, isa_label='isa', mode='intersection'):
    graph = nx.Graph(graph_component.graph)
    n = len(graph_component.graph)
    for ii, (subgraph_i, signature_i) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
        i = ii + n
        graph.add_node(i, label=signature_i, isa=list(subgraph_i.nodes()))
        for v in subgraph_i.nodes():
            graph.add_edge(i, v, label=isa_label, nesting=True)
        for jj, (subgraph_j, signature_j) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
            if jj > ii:
                j = jj + n
                if mode == 'intersection':
                    intersect = set(subgraph_i.nodes()) & set(subgraph_j.nodes())
                    if intersect:     # Not empty
                        graph.add_edge(i, j, label=len(intersect), isa=list(intersect))
                if mode == 'edge':
                    subgraph_i_all_edges = _get_edges(graph_component.graph, subgraph_i.nodes())
                    subgraph_j_all_edges = _get_edges(graph_component.graph, subgraph_j.nodes())
                    intersect = set(subgraph_i_all_edges) & set(subgraph_j_all_edges)
                    if intersect:     # Not empty
                        graph.add_edge(i, j, label=len(intersect), isa=list(intersect))

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
