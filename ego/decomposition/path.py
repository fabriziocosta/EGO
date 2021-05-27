#!/usr/bin/env python
"""Provides scikit interface."""


from toolz import curry
import networkx as nx
from ego.component import GraphComponent, serialize, get_subgraphs_from_edge_components


@curry
def path_decomposition(g, length=None, min_len=1, max_len=None):
    if length is not None:
        min_len = length
        max_len = length
    edge_components = []
    for u in g.nodes():
        for v in g.nodes():
            try:
                for path in nx.all_shortest_paths(g, source=u, target=v):
                    edge_component = set()
                    if len(path) >= min_len + 1 and len(path) <= max_len + 1:
                        for i, u in enumerate(path[:-1]):
                            w = path[i + 1]
                            edge_component.add((u, w))
                            #if u < w:
                            #    edge_component.add((u, w))
                            #else:
                            #    edge_component.add((w, u))
                    if edge_component:
                        edge_components.append(tuple(sorted(edge_component)))
            except Exception:
                pass
    return list(set(edge_components))


@curry
def decompose_path(graph_component, length=None, min_len=1, max_len=None):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = path_decomposition(subgraph, length=length, min_len=min_len, max_len=max_len)
        new_subgraphs = get_subgraphs_from_edge_components(graph_component.graph, components)
        new_signature = serialize(['path', length, min_len, max_len], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def pth(*args, **kargs): 
    return decompose_path(*args, **kargs)