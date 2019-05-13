#!/usr/bin/env python
"""Provides scikit interface."""


import networkx as nx
from collections import deque
from ego.component import GraphComponent
from ego.component import edge_to_node_components
from toolz import curry


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


def get_edges_bunch(graph, nbunch):
    g2 = nx.subgraph(graph, nbunch)
    ebunch = set()
    for u, v in g2.edges():
        if u < v:
            ebunch.add((u, v))
        else:
            ebunch.add((v, u))
    return ebunch


def context_component_decomposition(g, components=None, radius=1):
    neigh_edge_components = []
    for i, ci in enumerate(components):
        ebunch_i = get_edges_bunch(g, ci)
        sci = set(ci)
        neigh_edge_component = set()
        for j, cj in enumerate(components):
            if j != i:
                scj = set(cj)
                intersection = sci & scj
                for u in intersection:
                    nbunch = get_neighborhood_component(g, u, radius)
                    ebunch_u = get_edges_bunch(g, nbunch)
                    neigh_edge_component.update(ebunch_u)
        neigh_edge_components.append(neigh_edge_component - ebunch_i)
    neigh_edge_components = [c for c in neigh_edge_components if len(c) >= 2]
    return neigh_edge_components


@curry
def decompose_context(graph_component, radius=1):
    g = graph_component.graph
    nc = graph_component.node_components + edge_to_node_components(
        graph_component.edge_components)
    # if there are no components consider all edges
    if len(nc) == 0:
        nc = [set([u, v]) for u, v in g.edges()]
    ec = context_component_decomposition(g, nc, radius)
    gc = GraphComponent(
        graph=g,
        node_components=nc,
        edge_components=ec)
    return gc
