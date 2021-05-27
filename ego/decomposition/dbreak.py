#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
import numpy as np
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components
from itertools import combinations

def compute_effective_size(graph, size):
    if size is None:
        return len(graph)
    if size > 1:
        return size
    if 0 < size < 1:
        return int(len(graph)*size)
    assert False, 'Error on size: %s'%size

@curry
def decompose_break(graph_component, min_size=1, max_size=None, n_edges=1):
    new_subgraphs_list = []
    new_signatures_list = []
    # for each distinct pair of subgraphs
    for i, (g, signature_i) in enumerate(zip(graph_component.subgraphs, graph_component.signatures)):
        components = []
        effective_min_size = compute_effective_size(g,min_size)
        effective_max_size = compute_effective_size(g,max_size)
        for edges in combinations(g.edges(),n_edges):    
            gp = g.copy()
            for i,j in edges:
                gp.remove_edge(i, j)
            if nx.number_connected_components(gp) >= 2:
                for component in nx.connected_components(gp):
                    if effective_min_size <= len(component) <= effective_max_size: 
                        components.append(tuple(sorted(component)))
        components = set(components)
        for component in components:
            new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, [component])
            new_signature = serialize(['nbreak', min_size, n_edges], signature_i)
            new_subgraphs_list += new_subgraphs
            new_signatures_list += [new_signature]

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc

def brk(*args, **kargs): 
    return decompose_break(*args, **kargs)
