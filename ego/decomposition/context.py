#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
from collections import deque
from ego.component import GraphComponent, serialize, get_subgraphs_from_edge_components


def radius_rooted_breadth_first(graph, root, radius=1):
    dist_list = {}
    visited = set()  # use a set as we can end up exploring few nodes
    # q is the queue containing the frontier to be expanded in the BFV
    q = deque()
    q.append(root)
    # the map associates to each vertex id the distance from the root
    dist = {}
    dist[root] = 0
    visited.add(root)
    # add vertex at distance 0
    dist_list[0] = list()
    dist_list[0].append(root)
    while len(q) > 0:
        # extract the current vertex
        u = q.popleft()
        d = dist[u] + 1
        if d <= radius:
            # iterate over the neighbors of the current vertex
            for v in graph.neighbors(u):
                if v not in visited:
                    dist[v] = d
                    visited.add(v)
                    q.append(v)
                    if d not in dist_list:
                        dist_list[d] = list()
                    dist_list[d].append(v)
    return dist_list


def get_neighborhood_component(graph, root, radius=1):
    ns = radius_rooted_breadth_first(graph, root, radius)
    nbunch = set()
    for r in ns:
        nbunch.update(ns[r])
    return nbunch


def get_neighborhood(graph, component, radius=1):
    nbunch = set()
    for u in component:
        nbunch.update(get_neighborhood_component(graph, u, radius=radius))
    return nbunch


def get_edges_bunch(graph, nbunch):
    g2 = nx.subgraph(graph, nbunch)
    ebunch = set()
    for u, v in g2.edges():
        if u < v:
            ebunch.add((u, v))
        else:
            ebunch.add((v, u))
    return ebunch


def get_neighborhood_edges(graph, component, radius=1):
    nbunch = get_neighborhood(graph, component, radius)
    ebunch = get_edges_bunch(graph, nbunch)
    return ebunch


def context_component_decomposition(graph, subgraph=None, radius=1):
    component = set(subgraph.nodes())
    component_edges = get_edges_bunch(graph, component)
    neighbor_edges = get_neighborhood_edges(graph, component, radius)
    context_edges = neighbor_edges - component_edges
    context_edges = list(context_edges)
    return context_edges


@curry
def decompose_context(graph_component, radius=1):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        context_edges = context_component_decomposition(graph_component.graph, subgraph, radius)
        new_subgraphs = get_subgraphs_from_edge_components(graph_component.graph, [context_edges])
        new_signature = serialize(['context', radius], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def cntx(*args, **kargs): 
    return decompose_context(*args, **kargs)
