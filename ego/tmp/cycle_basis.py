#!/usr/bin/env python
"""Provides scikit interface."""


import networkx as nx
from ego.component import GraphComponent
from ego.component import get_subgraphs_from_graph_component
from ego.decompose import accumulate


def get_cycle_basis_edges(g):
    def get_edges_from_cycle(cycle):
        for i, c in enumerate(cycle):
            j = (i + 1) % len(cycle)
            u, v = cycle[i], cycle[j]
            if u < v:
                yield u, v
            else:
                yield v, u

    ebunch = []
    cs = nx.cycle_basis(g)
    for c in cs:
        ebunch += list(get_edges_from_cycle(c))
    return ebunch


def edge_complement(g, ebunch):
    edge_set = set(ebunch)
    other_ebunch = [e for e in g.edges() if e not in edge_set]
    return other_ebunch


def edge_subgraph(g, ebunch):
    g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for u, v in ebunch:
        g2.add_edge(u, v)
        g2.edges[u, v].update(g.edges[u, v])
    return g2


def edge_complement_subgraph(g, ebunch):
    """Induce graph from edges that are not in ebunch."""
    g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for e in g.edges():
        if e not in ebunch:
            u, v = e
            g2.add_edge(u, v)
            g2.edges[u, v].update(g.edges[u, v])
    return g2


def cycle_basis_and_non_cycle_decomposition(g):
    cs = nx.cycle_basis(g)
    cycle_components = list(map(set, cs))
    cycle_ebunch = get_cycle_basis_edges(g)
    g2 = edge_complement_subgraph(g, cycle_ebunch)
    non_cycle_components = nx.connected_components(g2)
    non_cycle_components = [c for c in non_cycle_components if len(c) >= 2]
    non_cycle_components = list(map(set, non_cycle_components))
    return cycle_components, non_cycle_components


def cycle_basis_decomposition(g):
    res = cycle_basis_and_non_cycle_decomposition(g)
    cycle_components, non_cycle_components = res
    return cycle_components


def non_cycle_decomposition(g):
    res = cycle_basis_and_non_cycle_decomposition(g)
    cycle_components, non_cycle_components = res
    return non_cycle_components


def cycle_and_non_cycle_decomposition(g):
    res = cycle_basis_and_non_cycle_decomposition(g)
    cycle_components, non_cycle_components = res
    all_components = cycle_components + non_cycle_components
    return all_components


def decompose_non_cycles(graph_component):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if not subgraphs:
        subgraphs = [graph_component.graph]
    nc = accumulate(non_cycle_decomposition, subgraphs)
    gc = GraphComponent(
        graph=graph_component.graph, node_components=nc, edge_components=[])
    return gc


def decompose_cycles(graph_component):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if not subgraphs:
        subgraphs = [graph_component.graph]
    nc = accumulate(cycle_basis_decomposition, subgraphs)
    gc = GraphComponent(
        graph=graph_component.graph, node_components=nc, edge_components=[])
    return gc


def decompose_cycles_and_non_cycles(graph_component):
    subgraphs = get_subgraphs_from_graph_component(graph_component)
    if not subgraphs:
        subgraphs = [graph_component.graph]
    nc = accumulate(cycle_and_non_cycle_decomposition, subgraphs)
    gc = GraphComponent(
        graph=graph_component.graph,
        node_components=nc,
        edge_components=[])
    return gc
