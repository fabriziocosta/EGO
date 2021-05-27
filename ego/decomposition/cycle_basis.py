#!/usr/bin/env python
"""Provides scikit interface."""


import networkx as nx
from ego.component import GraphComponent, serialize, get_subgraphs_from_node_components


def get_edges_from_cycle(cycle):
    for i, c in enumerate(cycle):
        j = (i + 1) % len(cycle)
        u, v = cycle[i], cycle[j]
        if u < v:
            yield u, v
        else:
            yield v, u


def get_cycle_basis_edges(g):
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
    if nx.is_directed(g):
        g2 = nx.DiGraph()
    else:
        g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for u, v in ebunch:
        g2.add_edge(u, v)
        g2.edges[u, v].update(g.edges[u, v])
    return g2


def edge_complement_subgraph(g, ebunch):
    """Induce graph from edges that are not in ebunch."""
    if nx.is_directed(g):
        g2 = nx.DiGraph()
    else:
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
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = non_cycle_decomposition(subgraph)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['non_cycles'], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def decompose_cycles(graph_component):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = cycle_basis_decomposition(subgraph)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['cycles'], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def decompose_cycles_and_non_cycles(graph_component):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        components = cycle_and_non_cycle_decomposition(subgraph)
        new_subgraphs = get_subgraphs_from_node_components(graph_component.graph, components)
        new_signature = serialize(['cycles_and_non_cycles'], signature)
        new_signatures = [new_signature] * len(new_subgraphs)
        new_subgraphs_list += new_subgraphs
        new_signatures_list += new_signatures

    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc

def cyc(*args, **kargs): 
    return decompose_cycles(*args, **kargs)

def ncyc(*args, **kargs): 
    return decompose_non_cycles(*args, **kargs)

def cycn(*args, **kargs): 
    return decompose_cycles_and_non_cycles(*args, **kargs)

