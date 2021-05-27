#!/usr/bin/env python
"""Provides scikit interface."""

import networkx as nx
from toolz import curry
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components
from collections import defaultdict


def get_rooted_dist2node_id_map(graph, root, cutoff=1):
    node_id2distance_map = nx.single_source_shortest_path_length(graph, root, cutoff=cutoff)
    dist2node_id_map = defaultdict(list)
    for node_id, distance in node_id2distance_map.items():
        dist2node_id_map[distance].append(node_id)
    return dist2node_id_map


def get_dist2node_id_map(graph, cutoff=1):
    return {u: get_rooted_dist2node_id_map(graph, u, cutoff=cutoff) for u in graph.nodes()}


def get_neighborhoods(graph, dist_dict, max_radius=1):
    graph_neighborhoods = dict()
    for u in dist_dict:
        neighborhoods = defaultdict(list)
        nbunch = []
        for d in range(max_radius + 1):
            nbunch.extend(dist_dict[u][d])
            neighborhoods[d] = list(nbunch)
        graph_neighborhoods[u] = neighborhoods
    return graph_neighborhoods


def get_pairs(graph, min_radius, max_radius, min_distance, max_distance):
    cutoff = max(max_radius, max_distance) + 1
    dist_dict = get_dist2node_id_map(graph, cutoff=cutoff)
    graph_neighborhoods = get_neighborhoods(graph, dist_dict, max_radius=max_radius)
    components = []
    for u in graph.nodes():
        for d in range(min_distance, max_distance + 1):
            for v in dist_dict[u][d]:
                for ru in range(min_radius, max_radius + 1):
                    neigh_u = set(graph_neighborhoods[u][ru])
                    for rv in range(min_radius, max_radius + 1):
                        neigh_v = set(graph_neighborhoods[v][rv])
                        components.append(tuple(neigh_v.union(neigh_u)))
    return set(components)


@curry
def decompose_paired_neighborhoods(graph_component, radius=None, distance=None,
                                   min_radius=0, max_radius=1,
                                   min_distance=0, max_distance=0):
    if radius is not None:
        min_radius = max_radius = radius
    if distance is not None:
        min_distance = max_distance = distance
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = get_pairs(subgraph, min_radius, max_radius, min_distance, max_distance)
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
def decompose_neighborhood(graph_component, radius=None, min_radius=0, max_radius=1):
    return decompose_paired_neighborhoods(
        graph_component, radius=radius, distance=None, min_radius=min_radius, max_radius=max_radius, min_distance=0, max_distance=0)


def ngb(*args, **kargs): 
    return decompose_neighborhood(*args, **kargs)


def prdngb(*args, **kargs): 
    return decompose_paired_neighborhoods(*args, **kargs)

