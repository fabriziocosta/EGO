#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import scipy as sp
import toolz as tz
import networkx as nx
from collections import defaultdict
from scipy.sparse import csr_matrix
from ego.encode import node_encode_graph
from ego.encode import _make_encoder
from ego.encode import make_fragmenter_func
from ego.component import convert
from ego.component import get_subgraphs_from_graph_component
from ego.vectorize import set_feature_size

def _to_sparse_matrix(encoding, node_ids, feature_size):
    mtx = defaultdict(int)
    for c, i in zip(encoding, node_ids):
        mtx[c, i] += 1
    data, row, col = [], [], []
    for c, i in mtx:
        row.append(i)
        col.append(c)
        data.append(mtx[(c, i)])
    shape = (max(row) + 1, feature_size)
    data_matrix = csr_matrix((data, (row, col)), shape=shape, dtype=np.float64)
    return data_matrix

def graph_node_vectorize(graph, decomposition_funcs, preprocessors=None, nbits=16):
    data_matrix = node_vectorize(graph, decomposition_funcs, preprocessors, nbits)
    return data_matrix.sum(axis=0)

def node_vectorize(graph, decomposition_funcs, preprocessors=None, nbits=16):
    feature_size, bitmask = set_feature_size(nbits=nbits)
    return _node_vectorize(graph, decomposition_funcs, preprocessors, bitmask, feature_size)

def _node_vectorize(graph, decomposition_funcs, preprocessors, bitmask, feature_size):
    fragmenter_func = make_fragmenter_func(decomposition_funcs, preprocessors)
    encoding, node_ids = node_encode_graph(graph, fragmenter_func, bitmask)
    data_matrix = _to_sparse_matrix(encoding, node_ids, feature_size)
    return data_matrix

def get_distance(graph):
    n = nx.number_of_nodes(graph)
    dist_mtx = np.zeros((n,n))
    for i,u in enumerate(graph.nodes()):
        for j,v in enumerate(graph.nodes()):
            if u != v :
                #dist_mtx[i,j] = nx.resistance_distance(graph, u, v)
                dist_mtx[i,j] = nx.shortest_path_length(graph, u, v)
    return dist_mtx

def get_proximity(graph, sigma=1.):
    dist_mtx = get_distance(graph)
    prox_mtx = np.exp(-dist_mtx/sigma)
    return prox_mtx

def proximity_node_vectorize(graph_orig, decomposition_funcs, preprocessors=None, nbits=16, sigma=1.):
    graph = nx.convert_node_labels_to_integers(graph_orig)
    prox_mtx = csr_matrix(get_proximity(graph, sigma=sigma))
    nodes_mtx = node_vectorize(graph, decomposition_funcs, preprocessors, nbits) 
    return prox_mtx.dot(nodes_mtx)

def node_attributes_vectorize(graph, attribute_label='attributes'):
    vec_list = [graph.nodes[u].get(attribute_label, [1]) for u in graph.nodes()]
    vecs = csr_matrix(np.array(vec_list))
    return vecs

def _real_vectorize_single(graph, decomposition_funcs, preprocessors, bitmask, feature_size):
    data_matrix = _node_vectorize(graph, decomposition_funcs, preprocessors, bitmask, feature_size)
    vecs = node_attributes_vectorize(graph, attribute_label='attributes')
    # combine node data matrix with node_attributes
    vec = sp.dot(data_matrix.T, vecs).T
    row_vec = [r for r in vec]
    attributed_vec = sp.sparse.hstack(row_vec)
    return attributed_vec


def real_vectorize(graphs,
                   decomposition_funcs,
                   preprocessors=None,
                   nbits=14,
                   seed=1):
    """real_vectorize."""
    feature_size, bitmask = set_feature_size(nbits=nbits)
    attributed_vecs_list = [_real_vectorize_single(
        graph, decomposition_funcs, bitmask, feature_size)
        for graph in graphs]
    attributed_vecs_mtx = sp.sparse.vstack(attributed_vecs_list)
    return attributed_vecs_mtx


def _real_node_vectorize_single(graph,
                                node_id,
                                decomposition_funcs,
                                preprocessors,
                                bitmask,
                                feature_size,
                                label_prefix='_=_'):
    g = graph.copy()
    g.nodes[node_id]['label'] = label_prefix + g.nodes[node_id]['label']
    vec = _real_vectorize_single(g, decomposition_funcs, bitmask, feature_size)
    return vec


def real_node_vectorize(graphs,
                        decomposition_funcs,
                        preprocessors=None,
                        nbits=14,
                        seed=1):
    """real_node_vectorize."""
    feature_size, bitmask = set_feature_size(nbits=nbits)
    attributed_vecs_mtx_list = []
    for graph in graphs:
        attributed_vecs_list = [_real_node_vectorize_single(
            graph, node_id, decomposition_funcs, preprocessors, bitmask, feature_size)
            for node_id in graph.nodes()]
        attributed_vecs_mtx = sp.sparse.vstack(attributed_vecs_list)
        attributed_vecs_mtx_list.append(attributed_vecs_mtx)
    return attributed_vecs_mtx_list
