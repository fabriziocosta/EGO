#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
from collections import deque
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


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


def get_pairs(graph, radius=1, distance=3):
    components = []
    for u in graph.nodes():
        neigh_u = get_neighborhood_component(graph, u, radius)
        dist_list = radius_rooted_breadth_first(graph, u, radius=distance)
        if distance in dist_list:
            for v in dist_list[distance]:
                neigh_v = get_neighborhood_component(graph, v, radius)
                neigh_v.update(neigh_u)
                components.append(neigh_v)
    return components


@curry
def decompose_paired_neighborhoods(graph_component, radius=1, distance=3):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = get_pairs(subgraph, radius, distance)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        if distance == 0:
            new_signature = serialize(['neighborhood', radius], signature)
        else:
            new_signature = serialize(['paired_neighborhoods', radius, distance], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


@curry
def decompose_neighborhood(graph_component, radius=1):
    return decompose_paired_neighborhoods(
        graph_component, radius=radius, distance=0)
