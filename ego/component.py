#!/usr/bin/env python
"""
Provides the definition of GraphComponent and utility functions.

A GraphComponent component as a named tuple containing 3 fields:
a graph and 2 components.
A components is a list of component entities.
A component can be a node component or an edge component.
A node component is a set of node identifiers.
An edge component is a set of 2-tuples of node identifiers,
with the convention that the first node id is always smaller than the
second node id (to guarantee identity matches).
"""


import networkx as nx
import collections

GraphComponent = collections.namedtuple(
    'GraphComponent', 'graph node_components edge_components')


def make_abstraction_graph(node_components):
    """Build the graph associated to a given node_components.

    A node is added for each component. An edge is added
    if two component share at least an element.
    The node label contains the list of elements in the component.
    The edge label contains the number of elements shared by the
    endpoints.

    Parameters
    ----------
    node_components : list[set(node ids)]
        The node_components.

    Returns
    -------
    graph : Networkx undirected graph
        The graph associated to the node_components.
    """
    graph = nx.Graph()
    for i, node_set in enumerate(node_components):
        graph.add_node(i + 1, label=list(node_set))
        for j, other_node_set in enumerate(node_components[:i]):
            intersect = node_set & other_node_set
            if intersect:     # Not empty
                graph.add_edge(i + 1, j + 1, label=len(intersect))
    return graph


def get_node_components_from_abstraction_graph(graph):
    """Extract the set of node ids from each node label in the input graph.

    Parameters
    ----------
    graph : Networkx undirected graphs
        Abstract graph with set of node ids as labels.

    Returns
    -------
    node_components : list[set(node ids)]
        A list of sets of node ids.
    """
    components = []
    for u in graph.nodes():
        components.append(set(graph.nodes[u]['label']))
    return components


def get_subgraphs_from_node_components(graph, node_components):
    """Build subgraphs from node sets.

    Parameters
    ----------
    node_components : list[set(node ids)]
        The node_components.

    Returns
    -------
    list of graphs : list of Networkx undirected graphs
        Each subgraph associated to each node_component.
    """
    graphs = []
    if node_components:
        for node_component in node_components:
            graphs.append(nx.subgraph(graph, node_component))
    return graphs


def get_subgraphs_from_abstraction_graph(graph, abstract_graph):
    node_components = get_node_components_from_abstraction_graph(abstract_graph)
    return get_subgraphs_from_node_components(graph, node_components)


def edge_to_node_component(edge_component):
    component = set()
    for u, v in edge_component:
        component.add(u)
        component.add(v)
    return component


def edge_to_node_components(edge_components):
    node_components = []
    for edge_component in edge_components:
        node_components.append(edge_to_node_component(edge_component))
    return node_components


def edge_subgraph(graph, edge_component):
    subgraph = nx.Graph()
    for u in edge_to_node_component(edge_component):
        subgraph.add_node(u)
        subgraph.nodes[u].update(graph.nodes[u])
    for u, v in edge_component:
        subgraph.add_edge(u, v)
        subgraph.edges[u, v].update(graph.edges[u, v])
    return subgraph


def get_subgraphs_from_edge_components(graph, edge_components):
    subgraphs = []
    if edge_components:
        for edge_component in edge_components:
            subgraphs.append(edge_subgraph(graph, edge_component))
    return subgraphs


def get_subgraphs_from_graph_component(graph_component):
    g = graph_component.graph
    nc = graph_component.node_components
    ec = graph_component.edge_components
    node_subgraphs = get_subgraphs_from_node_components(g, nc)
    edge_subgraphs = get_subgraphs_from_edge_components(g, ec)
    subgraphs = node_subgraphs + edge_subgraphs
    return subgraphs


def convert(graph):
    """Convert a graph to a GraphComponent namedtuple.

    Parameters
    ----------
    graph : Networkx undirected graph
        A graph.

    Returns
    -------
    GraphComponent : namedtuple
        GraphComponent(graph=graph, node_components=[], edge_components=[]).

    """
    # use instead a components_list made of pairs of node_components
    # and edge_components.
    # One can then use a reduce to join multiple components into one.
    gc = GraphComponent(graph=graph, node_components=[], edge_components=[])
    return gc
