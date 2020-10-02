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


def preprocess_distances(g, distance):
    near_sets = dict()
    dist_sets = dict()
    for u in g.nodes():
        ns = radius_rooted_breadth_first(g, u, radius=distance)
        near_set = set()
        for r in ns:
            if r != distance:
                near_set.update(ns[r])
        near_sets[u] = near_set
        if distance in ns:
            dist_sets[u] = set(ns[distance])
        else:
            dist_sets[u] = set()
    return near_sets, dist_sets


def pair_decomposition(graph, subgraphs, distance=1):
    near_sets, dist_sets = preprocess_distances(graph, distance)
    components = []
    # for each distinct pair of subgraphs
    for i, g in enumerate(subgraphs):
        g_node_set = set(g.nodes())
        g_near_set = set()
        g_dist_set = set()
        for u in g_node_set:
            g_near_set.update(near_sets[u])
            g_dist_set.update(dist_sets[u])
        g_dist_set = g_dist_set.difference(g_near_set)

        if len(g_dist_set) > 0:
            for j, m in enumerate(subgraphs):
                m_node_set = set(m.nodes())

                # if there is at least one node in m that is at the desired
                # distance and if there is no node in m that has a smaller
                # distance then make a component from the union of the nodes
                # for each node in g consider the set A of all nodes at
                # distance less than max and the set B of all nodes at distance
                # equal to max
                # consider the intersection of A with nodes in m:
                # this must be empty
                # consider the intersection of B with nodes in m:
                # this must be not empty
                near_intr_size = len(g_near_set.intersection(m_node_set))
                dist_intr_size = len(g_dist_set.intersection(m_node_set))
                condition = near_intr_size == 0 and dist_intr_size != 0
                if condition:
                    component = g_node_set.union(m_node_set)
                    components.append(component)
    return components


@curry
def decompose_pair(graph_component, distance=1):
    new_subgraphs_list = []
    new_signatures_list = []
    components_memory = set()
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = pair_decomposition(
            graph_component.graph, graph_component.subgraphs, distance=distance)
        new_components = []
        for component in components:
            c = tuple(component)
            if c not in components_memory:
                new_components.append(component)
                components_memory.add(c)
        if new_components:
            new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, new_components)
            new_signature = serialize(['pair', distance], signature)
            new_signatures = [new_signature] * len(new_subgraphs)
            new_subgraphs_list += new_subgraphs
            new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc
