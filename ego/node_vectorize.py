#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import scipy as sp
import toolz as tz
import networkx as nx
from itertools import combinations
from collections import defaultdict
from scipy.sparse import csr_matrix
from ego.encode import node_encode_graph
from ego.encode import _make_encoder
from ego.encode import make_fragmenter_func
from ego.encode import fast_hash
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

def _node_vectorize(graph, decomposition_funcs, preprocessors, bitmask, feature_size):
    fragmenter_func = make_fragmenter_func(decomposition_funcs, preprocessors)
    encoding, node_ids = node_encode_graph(graph, fragmenter_func, bitmask)
    data_matrix = _to_sparse_matrix(encoding, node_ids, feature_size)
    return data_matrix

def node_vectorize(graph, decomposition_funcs, preprocessors=None, nbits=16):
    feature_size, bitmask = set_feature_size(nbits=nbits)
    return _node_vectorize(graph, decomposition_funcs, preprocessors, bitmask, feature_size)

def get_distance(graph, weight=None, type_of='shortest'):
    if type_of == 'shortest':
        dist_func = nx.shortest_path_length
    elif type_of == 'resistance':
        dist_func = nx.resistance_distance
    else:
        raise Exception('Unknown type of distance:%s'%type_of)
    n = nx.number_of_nodes(graph)
    dist_mtx = np.zeros((n,n))
    for i,u in enumerate(graph.nodes()):
        for j,v in enumerate(graph.nodes()):
            if u != v :
                dist_mtx[i,j] = dist_func(graph, u, v, weight=weight)
    return dist_mtx

def get_proximity(graph, effective_radius=1, weight=None, type_of='shortest', cutoff_effective_radius_factor=2):
    if effective_radius == 0:
        n = nx.number_of_nodes(graph)
        return np.eye(n,n)
    if effective_radius is None: 
        effective_radius = nx.diameter(graph) / 4
    dist_mtx = get_distance(graph, weight=weight, type_of=type_of)
    prox_mtx = np.exp( - dist_mtx / effective_radius)
    prox_mtx[dist_mtx >= cutoff_effective_radius_factor * effective_radius] = 0
    return prox_mtx

def build_node_proximity_data_matrix(nodes_mtx, data_matrix, nbits, max_num_node_features=1):
    feature_size, bitmask = set_feature_size(nbits=nbits)
    data, row, col = [], [], []
    for node_id, (node_vec, row_vec) in enumerate(zip(nodes_mtx, data_matrix)):
        all_combinations = [sorted(comb) for combinations_order in range(1, max_num_node_features + 1) for comb in combinations(node_vec.indices, combinations_order)]
        for node_feature_id_combinations in all_combinations:
            node_feature_id_combinations = list(node_feature_id_combinations)
            for remote_feature_id, remote_feature_val in zip(row_vec.indices, row_vec.data):
                new_feature_id = fast_hash(node_feature_id_combinations+[remote_feature_id], bitmask=bitmask)
                row.append(node_id)
                col.append(new_feature_id)
                data.append(remote_feature_val)
    shape = (max(row) + 1, feature_size)
    new_data_matrix = csr_matrix((data, (row, col)), shape=shape, dtype=np.float64) 
    return new_data_matrix

def node_attributes_vectorize(graph, attribute_label='attributes'):
    vec_list = [graph.nodes[u].get(attribute_label, [0]) for u in graph.nodes()]
    vecs = csr_matrix(np.array(vec_list))
    return vecs

def proximity_node_vectorize(graph_orig, decomposition_funcs, preprocessors=None, nbits=16, effective_radius=1, cutoff_effective_radius_factor=2, weight=None, type_of='shortest', attribute_label=None, return_node_mtx=False):
    graph = nx.convert_node_labels_to_integers(graph_orig)
    prox_mtx = csr_matrix(get_proximity(graph, effective_radius=effective_radius, weight=weight, type_of=type_of, cutoff_effective_radius_factor=cutoff_effective_radius_factor))
    nodes_mtx = node_vectorize(graph, decomposition_funcs, preprocessors, nbits) 
    
    if attribute_label is not None:
        attribute_nodes_mtx = node_attributes_vectorize(graph, attribute_label=attribute_label)
        nodes_mtx = sp.sparse.hstack([attribute_nodes_mtx, nodes_mtx])
        nodes_mtx = csr_matrix(nodes_mtx)
        
    data_matrix = prox_mtx.dot(nodes_mtx)
    
    if return_node_mtx:
        return nodes_mtx, data_matrix
    else:
        return data_matrix

def node_proximity_node_vectorize(graph_orig, decomposition_funcs, preprocessors=None, nbits=16, effective_radius=1, cutoff_effective_radius_factor=2, max_num_node_features=1, weight=None, type_of='shortest', attribute_label=None):
    nodes_mtx, data_matrix = proximity_node_vectorize(graph_orig, decomposition_funcs, preprocessors=preprocessors, nbits=nbits, effective_radius=effective_radius, cutoff_effective_radius_factor=cutoff_effective_radius_factor, weight=weight, type_of=type_of, attribute_label=attribute_label, return_node_mtx=True)
    node_proximity_data_matrix = build_node_proximity_data_matrix(nodes_mtx, data_matrix, nbits, max_num_node_features)
    return node_proximity_data_matrix

def _graph_node_vectorize(graph, decomposition_funcs, preprocessors=None, nbits=16, effective_radius=1, cutoff_effective_radius_factor=2, max_num_node_features=1, weight=None, type_of='shortest', attribute_label=None):
    data_matrix = node_proximity_node_vectorize(graph, decomposition_funcs, preprocessors, nbits, effective_radius, cutoff_effective_radius_factor, max_num_node_features, weight, type_of, attribute_label)
    vec = csr_matrix(csr_matrix.sum(data_matrix, axis=0))
    return vec

def graph_node_vectorize(graphs, decomposition_funcs, preprocessors=None, nbits=16, effective_radius=1, cutoff_effective_radius_factor=2, max_num_node_features=1, weight=None, type_of='shortest', attribute_label=None):
    data_list = [_graph_node_vectorize(graph, decomposition_funcs, preprocessors, nbits, effective_radius, cutoff_effective_radius_factor, max_num_node_features, weight, type_of, attribute_label) for graph in graphs]
    data_matrix = sp.sparse.vstack(data_list)
    return data_matrix
