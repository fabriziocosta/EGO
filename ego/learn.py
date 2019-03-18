#!/usr/bin/env python
"""Provides wrapper for optimizer."""

from ego.vectorize import set_feature_size
from ego.encode import make_encoder
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
import GraphOptimizer.treeinterpreter as ti
from ego.vectorize import vectorize_graphs
from sklearn.linear_model import SGDClassifier
import networkx as nx
from collections import defaultdict


def compute_features_scores(graphs, targets, encoding_func=None, feature_size=None):
    x = vectorize_graphs(graphs, encoding_func, feature_size)
    estimator = SGDClassifier(penalty='elasticnet', tol=1e-3)
    fs = RFECV(estimator, step=.1, cv=3)
    fs.fit(x, targets)
    fs.estimator_.decision_function(fs.transform(x)).reshape(-1)
    importances = fs.inverse_transform(fs.estimator_.coef_).reshape(-1)
    intercept = fs.estimator_.intercept_[0]
    importance_dict = dict(enumerate(importances))
    return importance_dict, intercept


def node_importance(graph, encoding_func, importance_dict, intercept):
    from collections import defaultdict
    node_score_dict = defaultdict(float)
    codes, fragments = encoding_func(graph)
    n_fragments = float(len(fragments))
    for code, fragment in zip(codes, fragments):
        n_nodes_in_fragment = float(len(fragment))
        for u in fragment.nodes():
            node_score_dict[u] += (importance_dict[code] +
                                   intercept / n_fragments) / n_nodes_in_fragment
    return node_score_dict


def node_and_edge_importance(graph, encoding_func, importance_dict, intercept):
    node_scores = defaultdict(float)
    edge_scores = defaultdict(float)
    codes, fragments = encoding_func(graph)
    n_fragments = float(len(fragments))
    for code, fragment in zip(codes, fragments):
        n_nodes_in_fragment = float(nx.number_of_nodes(fragment))
        n_edges_in_fragment = float(nx.number_of_edges(fragment))
        for u in fragment.nodes():
            node_scores[u] += (importance_dict[code] +
                               intercept / n_fragments) / n_nodes_in_fragment
        for u, v in fragment.edges():
            if u > v:
                u, v = v, u
            edge_scores[(u, v)] += (importance_dict[code] +
                                    intercept / n_fragments) / n_edges_in_fragment
    return node_scores, edge_scores


def make_node_scoring_func(graphs, targets, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1):
    feature_size, bitmask = set_feature_size(nbits=nbits)
    encoding_func = make_encoder(
        decomposition_funcs,
        preprocessors=preprocessors,
        bitmask=bitmask,
        seed=seed)

    importance_dict, intercept = compute_features_scores(
        graphs, targets, encoding_func=encoding_func, feature_size=feature_size)

    def node_scoring_func(graph):
        node_scores, edge_scores = node_and_edge_importance(
            graph, encoding_func, importance_dict, intercept)
        return node_scores

    return node_scoring_func
