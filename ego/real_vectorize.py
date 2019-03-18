#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import scipy as sp
import toolz as tz
from collections import defaultdict
from scipy.sparse import csr_matrix
from ego.encode import node_encode_graph
from ego.component import convert
from ego.component import get_subgraphs_from_graph_component
from ego.vectorize import set_feature_size


def to_sparse_matrix(encoding, node_ids, feature_size):
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


def real_vectorize_single(graph, decomposition_funcs, bitmask, feature_size):
    encoding, node_ids = node_encode_graph(graph, tz.compose(
        get_subgraphs_from_graph_component, decomposition_funcs, convert),
        bitmask=bitmask)
    data_matrix = to_sparse_matrix(encoding, node_ids, feature_size)
    vecs = csr_matrix(np.array([graph.nodes[u]['vec'] for u in graph.nodes()]))
    vec = sp.dot(data_matrix.T, vecs).T
    row_vec = [r for r in vec]
    vvec = sp.sparse.hstack(row_vec)
    return vvec


def real_vectorize(graphs, decomposition_funcs, preprocessors=None,
                   nbits=10, seed=1):
    feature_size, bitmask = set_feature_size(nbits=nbits)
    return sp.sparse.vstack([real_vectorize_single(
        graph, decomposition_funcs, bitmask, feature_size)
        for graph in graphs])
