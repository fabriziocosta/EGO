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


def get_node_neighborhood_component(graph, root, radius=1):
    neighbors_dict = radius_rooted_breadth_first(graph, root, radius)
    nbunch = set()
    for radius_key in neighbors_dict:
        nbunch.update(neighbors_dict[radius_key])
    return nbunch


def get_component_neighborhood_component(g, subgraph, radius=1):
    node_component = set()
    for u in subgraph.nodes():
        nbunch = get_node_neighborhood_component(g, u, radius)
        node_component.update(nbunch)
    return node_component


@curry
def decompose_dilatate(graph_component, radius=1):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        component = get_component_neighborhood_component(graph_component.graph, subgraph, radius)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, [component])
        new_signature = serialize(['dilatate', radius], signature)
        new_subgraphs_list += new_subgraphs
        new_signatures_list.append(new_signature)

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc
