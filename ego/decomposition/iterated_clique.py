#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
from ego.component import GraphComponent, serialize, get_node_components_from_abstraction_graph
from ego.component import make_abstraction_graph, edge_to_node_components


def make_cliq_graph(g, min_size=2, max_size=None):
    if max_size is not None:
        e_size = min(max_size, len(g))
    else:
        e_size = len(g)
    cliqs_list = list(map(set, filter(
        lambda c: min_size <= len(c) < e_size, nx.find_cliques(g))))
    g2 = make_abstraction_graph(cliqs_list)
    return g2


def transitive_reference_update(g1, g2):
    # g1 and g2 have node attribute: contains
    # we want to update the reference to node ids in g2, using g1
    # if g2 has node 0 that contains nodes 2 and 5
    # and g1 has node 2 containing [1,3,5] and node 5 containing [3,8,9]
    # the node 0 in g2 should contain [1,3,5,8,9] i.e. the set union
    g3 = g2.copy()
    for u2 in g2.nodes():
        contained = g2.nodes[u2]['label']
        new_contained = set()
        for u1 in contained:
            original_contained = g1.nodes[u1]['label']
            new_contained.update(original_contained)
        g3.nodes[u2]['label'] = list(new_contained)
    return g3


def get_components_iterated_clique_decomposition(
        graph, components=None, n_iter=1, min_n_iter=0, min_size=2, max_size=None):
    gs = []
    g2 = make_abstraction_graph(components)
    if min_n_iter == 0:
        gs.append(g2.copy())
    for i in range(n_iter):
        g3 = make_cliq_graph(g2, min_size=min_size, max_size=max_size)
        g4 = transitive_reference_update(g2, g3)
        if i + 1 >= min_n_iter:
            gs.append(g4.copy())
        g2 = g4
    return gs


def iterated_clique_decomposition(
        g, components=None, n_iter=1, min_n_iter=0, min_size=2, max_size=None):
    graphs = get_components_iterated_clique_decomposition(
        g, components, n_iter, min_n_iter=min_n_iter, min_size=min_size, max_size=max_size)
    components = []
    for graph in graphs:
        components += get_node_components_from_abstraction_graph(graph)
    return components


@curry
def decompose_iterated_clique(
        graph_component, n_iter=1, min_n_iter=0, min_size=2, max_size=None):
    components = graph_component.node_components
    components += edge_to_node_components(graph_component.edge_components)
    nc = iterated_clique_decomposition(
        graph_component.graph,
        components=components,
        n_iter=n_iter,
        min_n_iter=min_n_iter,
        min_size=min_size,
        max_size=max_size)
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=nc,
        edge_components=[],
        signature=serialize(['iterated_clique', n_iter, min_n_iter, min_size, max_size],
                            graph_component.signature))
    return gc

def itrclq(*args, **kargs): 
    return decompose_iterated_clique(*args, **kargs)    
    
