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


@curry
def decompose_pair_binary(graph_component_first, graph_component_second, distance=1, keep_second_component=True):
    new_subgraphs_list = []
    new_signatures_list = []

    near_sets, dist_sets = preprocess_distances(
        graph_component_first.graph, distance)

    # for each distinct pair of subgraphs
    for g, signature_g in zip(
            graph_component_first.subgraphs, graph_component_first.signatures):
        g_node_set = set(g.nodes())
        g_near_set = set()
        g_dist_set = set()
        for u in g_node_set:
            g_near_set.update(near_sets[u])
            g_dist_set.update(dist_sets[u])
        g_dist_set = g_dist_set.difference(g_near_set)

        if len(g_dist_set) > 0:
            for m, signature_m in zip(
                    graph_component_second.subgraphs, graph_component_second.signatures):
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
                    if keep_second_component:
                        component = g_node_set.union(m_node_set)
                    else:
                        component = g_node_set
                    new_subgraph = get_subgraphs_from_node_components(
                        graph_component_first.graph, [component])
                    new_subgraphs_list += new_subgraph
                    new_signature = serialize(
                        ['pair_binary', distance, keep_second_component], signature_g, signature_m)
                    new_signatures_list += new_signature

    gc = GraphComponent(
        graph=graph_component_first.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc
