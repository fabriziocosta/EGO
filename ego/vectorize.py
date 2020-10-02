#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from ego.encode import make_encoder

_bitmask_ = 4294967295


def convert_dict_to_sparse_matrix(feature_rows, feature_size):
    if len(feature_rows) == 0:
        # case of empty feature set for all instances
        return csr_matrix((1, feature_size))
    data, row, col = [], [], []
    for i, feature_row in enumerate(feature_rows):
        if len(feature_row) == 0:
            # case of empty feature set for a specific instance
            row.append(i)
            col.append(0)
            data.append(0)
        else:
            for feature in feature_row:
                row.append(i)
                col.append(feature)
                data.append(feature_row[feature])
    shape = (max(row) + 1, feature_size)
    data_matrix = csr_matrix((data, (row, col)),
                             shape=shape, dtype=np.float64)
    return data_matrix


def to_dict(codes, fragments):
    row = defaultdict(int)
    for code, fragment in zip(codes, fragments):
        row[code] += fragment.number_of_nodes()
    return row


def set_feature_size(nbits=14):
    bitmask = pow(2, nbits) - 1
    feature_size = bitmask + 2
    return feature_size, bitmask


def vectorize_graphs(graphs, encoding_func=None, feature_size=None):
    feature_rows = list()
    for graph in graphs:
        g_codes, g_frags = encoding_func(graph)
        g_dict = to_dict(g_codes, g_frags)
        feature_rows.append(g_dict)
    mtx = convert_dict_to_sparse_matrix(feature_rows, feature_size)
    return mtx


def vectorize(graphs, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1):
    feature_size, bitmask = set_feature_size(nbits=nbits)
    encoding_func = make_encoder(
        decomposition_funcs,
        preprocessors=preprocessors,
        bitmask=bitmask,
        seed=seed)
    mtx = vectorize_graphs(
        graphs, encoding_func=encoding_func, feature_size=feature_size)
    return mtx


def hash_graph(graph, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1):
    feature_size, bitmask = set_feature_size(nbits=nbits)
    encoding_func = make_encoder(
        decomposition_funcs,
        preprocessors=preprocessors,
        bitmask=bitmask,
        seed=seed)
    codes, fragments = encoding_func(graph)
    dat = tuple(sorted(codes))
    return int(hash(dat) & bitmask) + 1


def get_feature_dict(graphs, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1, return_counts=False):
    feature_size, bitmask = set_feature_size(nbits=nbits)
    encoding_func = make_encoder(
        decomposition_funcs,
        preprocessors=preprocessors,
        bitmask=bitmask,
        seed=seed)
    feature_dict = dict()
    feature_counts_dict = defaultdict(int)
    for graph in graphs:
        codes, fragments = encoding_func(graph)
        feature_dict.update(zip(codes, fragments))
        for code in codes:
            feature_counts_dict[code] += 1
    if return_counts:
        return feature_dict, feature_counts_dict
    else:
        return feature_dict
        

def get_feature_set(graphs, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1):
    feature_size, bitmask = set_feature_size(nbits=nbits)
    encoding_func = make_encoder(
        decomposition_funcs,
        preprocessors=preprocessors,
        bitmask=bitmask,
        seed=seed)
    feature_set = set()
    for graph in graphs:
        codes, fragments = encoding_func(graph)
        feature_set.add(codes)
    return feature_set
