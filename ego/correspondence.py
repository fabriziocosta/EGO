#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ego.abstraction.abstract import make_abstract_graph
from ego.real_vectorize import real_node_vectorize
from eden.display import draw_graph, map_labels_to_colors
from ego.setup import do_decompose, decompose_nodes_and_edges, decompose_neighborhood, decompose_context, decompose_cycles, decompose_non_cycles


def stable(rankings, list_a, list_b):
    partners = dict((a, (rankings[(a, 1)], 1)) for a in list_a)
    # whether the current pairing (given by `partners`) is stable
    is_stable = False
    while is_stable is False:
        is_stable = True
        for b in list_b:
            is_paired = False  # whether b has a pair which b ranks <= to n
            for n in range(1, len(list_b) + 1):
                a = rankings[(b, n)]
                a_partner, a_n = partners[a]
                if a_partner == b:
                    if is_paired:
                        is_stable = False
                        partners[a] = (rankings[(a, a_n + 1)], a_n + 1)
                    else:
                        is_paired = True
    stable_list = sorted((a, b) for (a, (b, n)) in partners.items())
    return stable_list


def get_pos_from_abstract_graph(abstract_graph):
    base_nodes = [u for u in abstract_graph.nodes() if 'isa' not in abstract_graph.nodes[u]]
    graph = nx.subgraph(abstract_graph, base_nodes)
    # compute the position of non abstract vertices
    pos = nx.kamada_kawai_layout(graph)
    # extract the map abstract node -> base nodes list
    abstract_map = {u: abstract_graph.nodes[u]['isa'] for u in abstract_graph.nodes() if 'isa' in abstract_graph.nodes[u]}
    # compute the pos of an abstract node as the average of the pos of the nodes it abstracts
    abstract_pos = {k: np.array([pos[val] for val in vals]).mean(axis=0) for k, vals in abstract_map.items()}
    # add the pos of the abstract nodes
    pos.update(abstract_pos)
    return pos


def get_distance_correlation(id_A, id_B, map_from_A_to_B):
    dist_A = nx.single_source_shortest_path_length(graph_A, id_A)
    dist_B = nx.single_source_shortest_path_length(graph_B, id_B)

    A_d_vec = []
    B_d_vec = []
    for nid_A, d_A in dist_A.items():
        nid_B = map_from_A_to_B[nid_A]
        d_B = dist_B[nid_B]
        A_d_vec.append(d_A)
        B_d_vec.append(d_B)
    A_d_vec = np.array(A_d_vec)
    B_d_vec = np.array(B_d_vec)
    r = np.corrcoef(A_d_vec, B_d_vec)[0, 1]
    return r


def map_first_onto_second(GA, GB, pos=None, decomposition_function=None):
    if decomposition_function is None:
        decompose_context1_of_nodes_edges_neighborhood1 = do_decompose(decompose_nodes_and_edges, decompose_neighborhood(radius=1), compose_function=decompose_context(radius=1))
        decompose_context2_of_nodes_edges_neighborhood1 = do_decompose(decompose_nodes_and_edges, decompose_neighborhood(radius=1), compose_function=decompose_context(radius=2))
        decomposition_function = do_decompose(decompose_cycles, decompose_neighborhood(max_radius=2), decompose_context1_of_nodes_edges_neighborhood1, decompose_context2_of_nodes_edges_neighborhood1)

    graph_A = GA.copy()
    graph_B = GB.copy()
    n_A = graph_A.number_of_nodes()
    n_B = graph_B.number_of_nodes()
    if n_A < n_B:
        n = n_B - n_A
        for i in range(n):
            graph_A.add_node(n_A + i, label='-')
    elif n_B < n_A:
        n = n_A - n_B
        for i in range(n):
            graph_B.add_node(n_B + i, label='-')

    mapping_A = {i: 'A%03d' % (i + 1) for i in range(graph_A.number_of_nodes())}
    mapping_A_inv = {v: k for k, v in mapping_A.items()}
    for u in graph_A.nodes():
        graph_A.nodes[u]['map'] = mapping_A[u]

    mapping_B = {i: 'B%03d' % (i + 1) for i in range(graph_B.number_of_nodes())}
    mapping_B_inv = {v: k for k, v in mapping_B.items()}
    for u in graph_B.nodes():
        graph_B.nodes[u]['map'] = mapping_B[u]

    X_A, X_B = real_node_vectorize([nx.convert_node_labels_to_integers(graph_A), nx.convert_node_labels_to_integers(graph_B)], decomposition_funcs=decomposition_function)

    rankings = dict()
    for i in range(graph_A.number_of_nodes()):
        s = [-X_A[i].dot(X_B[j].T)[0, 0] for j in range(graph_B.number_of_nodes())]
        r = np.argsort(s)
        for n, rr in enumerate(r):
            a_id = mapping_A[i]
            b_id = mapping_B[rr]
            pref = n + 1
            rankings[(a_id, pref)] = b_id

    for i in range(graph_B.number_of_nodes()):
        s = [-X_B[i].dot(X_A[j].T)[0, 0] for j in range(graph_A.number_of_nodes())]
        r = np.argsort(s)
        for n, rr in enumerate(r):
            a_id = mapping_B[i]
            b_id = mapping_A[rr]
            pref = n + 1
            rankings[(a_id, pref)] = b_id

    ids_A = [graph_A.nodes[u]['map'] for u in graph_A.nodes()]
    ids_B = [graph_B.nodes[u]['map'] for u in graph_B.nodes()]
    match_A_to_B = stable(rankings, ids_A, ids_B)

    map_from_A_to_B = dict()
    for A, B in match_A_to_B:
        A_id = mapping_A_inv[A]
        B_id = mapping_B_inv[B]
        if graph_B.nodes[B_id]['label'] != '-':
            map_from_A_to_B[A_id] = B_id

    fixed = []
    pos_A = dict()
    pos_B = get_pos_from_abstract_graph(graph_B)
    if pos is not None:
        pos_B.update(pos)
    for A, B in match_A_to_B:
        A_id = mapping_A_inv[A]
        B_id = mapping_B_inv[B]
        if graph_B.nodes[B_id]['label'] != '-':
            fixed.append(A_id)
        pos_A[A_id] = (pos_B[B_id][0], pos_B[B_id][1])
    pos_A = nx.spring_layout(graph_A, pos=pos_A, fixed=fixed)

    for u in list(graph_A.nodes()):
        if graph_A.nodes[u]['label'] == '-':
            graph_A.remove_node(u)

    for u in list(graph_B.nodes()):
        if graph_B.nodes[u]['label'] == '-':
            graph_B.remove_node(u)

    return graph_A, pos_A, graph_B, pos_B, map_from_A_to_B


def is_abstract_graph(abstract_graph):
    abstract_nodes = [u for u in abstract_graph.nodes() if 'isa' in abstract_graph.nodes[u]]
    return len(abstract_nodes) > 0


def display_correspondences(GA, GB, decomposition_function=None, size=4):
    plt.figure(figsize=(4.2 * size, size))

    graph_A, pos_A, graph_B, pos_B, map_from_A_to_B = map_first_onto_second(GA, GB, decomposition_function=decomposition_function)
    colors = map_labels_to_colors([graph_A, graph_B])
    kwargs = dict(size=None, colormap='tab20c', vertex_size=80, edge_label=None, vertex_color_dict=colors, vmax=1, vmin=0, vertex_color='_label_', vertex_label=None, layout='spring', ignore_for_layout='nesting')
    plt.subplot(1, 4, 1)
    draw_graph(graph_A, pos=pos_A, **kwargs)
    plt.subplot(1, 4, 2)
    draw_graph(graph_B, pos=pos_B, **kwargs)

    graph_A_i, pos_A_i, graph_B_i, pos_B_i, map_from_A_to_B_i = map_first_onto_second(GB, GA, decomposition_function=decomposition_function)
    plt.subplot(1, 4, 3)
    draw_graph(graph_A_i, pos=pos_A_i, **kwargs)
    plt.subplot(1, 4, 4)
    draw_graph(graph_B_i, pos=pos_B_i, **kwargs)

    plt.show()

    if is_abstract_graph(GA) and is_abstract_graph(GB):
        graph_A_base_nodes = [u for u in graph_A.nodes() if 'isa' not in graph_A.nodes[u]]
        graph_A_base = nx.subgraph(graph_A, graph_A_base_nodes)
        graph_B_base_nodes = [u for u in graph_B.nodes() if 'isa' not in graph_B.nodes[u]]
        graph_B_base = nx.subgraph(graph_B, graph_B_base_nodes)

        graph_A_i_base_nodes = [u for u in graph_A_i.nodes() if 'isa' not in graph_A_i.nodes[u]]
        graph_A_i_base = nx.subgraph(graph_A_i, graph_A_i_base_nodes)
        graph_B_i_base_nodes = [u for u in graph_B_i.nodes() if 'isa' not in graph_B_i.nodes[u]]
        graph_B_i_base = nx.subgraph(graph_B_i, graph_B_i_base_nodes)

        plt.figure(figsize=(4.2 * size, size))

        plt.subplot(1, 4, 1)
        draw_graph(graph_A_base, pos=pos_A, **kwargs)
        plt.subplot(1, 4, 2)
        draw_graph(graph_B_base, pos=pos_B, **kwargs)

        plt.subplot(1, 4, 3)
        draw_graph(graph_A_i_base, pos=pos_A_i, **kwargs)
        plt.subplot(1, 4, 4)
        draw_graph(graph_B_i_base, pos=pos_B_i, **kwargs)

        plt.show()


def display_correspondences_with_abstraction(GA, GB, abstract_decomposition_function=None, decomposition_function=None):
    if abstract_decomposition_function is None:
        abstract_decomposition_function = do_decompose(decompose_cycles, decompose_non_cycles)
    if decomposition_function is None:
        decompose_context1_of_nodes_edges_neighborhood1 = do_decompose(decompose_nodes_and_edges, decompose_neighborhood(radius=1), compose_function=decompose_context(radius=1))
        decompose_context2_of_nodes_edges_neighborhood1 = do_decompose(decompose_nodes_and_edges, decompose_neighborhood(radius=1), compose_function=decompose_context(radius=2))
        decomposition_function = do_decompose(decompose_context1_of_nodes_edges_neighborhood1, decompose_context2_of_nodes_edges_neighborhood1, decompose_neighborhood(max_radius=2))

    AGA, AGB = make_abstract_graph([GA, GB], decomposition=abstract_decomposition_function)
    display_correspondences(AGA, AGB, decomposition_function=decomposition_function)


def display_multiple_correspondences(Gs, GB, decomposition_function=None, size=6, n_graphs_per_line=6):
    pos = get_pos_from_abstract_graph(GB)
    if decomposition_function is None:
        df1 = do_decompose(decompose_nodes_and_edges, decompose_neighborhood(radius=1), compose_function=decompose_context(radius=1))
        df2 = do_decompose(decompose_nodes_and_edges, decompose_neighborhood(radius=1), compose_function=decompose_context(radius=2))
        decomposition_function = do_decompose(df1, df2, decompose_cycles, decompose_neighborhood(max_radius=3))
    n = len(Gs)
    n_rows = n // n_graphs_per_line
    plt.figure(figsize=(n_graphs_per_line * size, size))
    for i in range(n):
        GA = Gs[i]
        graph_A, pos_A, graph_B, pos_B, map_from_A_to_B = map_first_onto_second(GA, GB, pos=pos, decomposition_function=decomposition_function)
        colors = map_labels_to_colors([graph_A, graph_B])
        kwargs = dict(size=None, colormap='tab20c', vertex_size=80, edge_label=None, vertex_color_dict=colors, vmax=1, vmin=0, vertex_color='_label_', vertex_label=None, layout='spring', ignore_for_layout='nesting')
        plt.subplot(n_rows + 1, n_graphs_per_line, i + 1)
        draw_graph(graph_A, pos=pos_A, **kwargs)
    plt.show()
