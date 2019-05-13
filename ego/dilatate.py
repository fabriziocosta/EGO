#!/usr/bin/env python
"""Provides scikit interface."""


from collections import deque
from toolz import curry
from ego.component import GraphComponent
from ego.component import edge_to_node_components


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


def get_node_neighborhood_component(graph, root, radius=1):
    neighbors_dict = radius_rooted_breadth_first(graph, root, radius)
    nbunch = set()
    for radius_key in neighbors_dict:
        nbunch.update(neighbors_dict[radius_key])
    return nbunch


def get_component_neighborhood_component(g, component, radius=1):
    node_component = set()
    for u in component:
        nbunch = get_node_neighborhood_component(g, u, radius)
        node_component.update(nbunch)
    return node_component


@curry
def decompose_dilatate(graph_component, radius=1):
    g = graph_component.graph
    nc = graph_component.node_components + edge_to_node_components(
        graph_component.edge_components)
    # if there are no components consider all edges
    if len(nc) == 0:
        nc = [set([u, v]) for u, v in g.edges()]
    node_components = []
    for component in nc:
        new_component = get_component_neighborhood_component(
            g, component, radius)
        node_components.append(new_component)
    gc = GraphComponent(
        graph=g, node_components=node_components, edge_components=[])
    return gc
