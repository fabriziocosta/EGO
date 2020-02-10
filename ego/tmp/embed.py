#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import networkx as nx
from sklearn import manifold
from ego.vectorize import get_feature_dict
from ego.vectorize import set_feature_size
from ego.encode import make_encoder
import matplotlib.pyplot as plt


def embed_graph_mds(graph, n_components=2, weight='len'):
    p = dict(nx.shortest_path_length(graph, weight=weight))
    mapper = {u: i for i, u in enumerate(graph.nodes())}
    dissimilarity = np.zeros((len(graph), len(graph)))
    for i in graph.nodes():
        ii = mapper[i]
        for j in graph.nodes():
            jj = mapper[j]
            if p[i].get(j, None) is not None:
                dissimilarity[ii, jj] = p[i][j]
    mds = manifold.MDS(
        n_components=n_components,
        dissimilarity="precomputed")
    x = mds.fit(dissimilarity).embedding_
    return x


def canonical_direction(x):
    u, s, vh = np.linalg.svd(x, full_matrices=True)
    z = np.dot(x, vh.T)
    return z


def scale_largest_dim(x):
    xmin = np.min(x, axis=0).reshape(-1)
    xmax = np.max(x, axis=0).reshape(-1)
    xlen = np.absolute(xmax - xmin).max()
    x = (x - xmin) / xlen
    return x


def embed(graph, n_components=2, weight='len'):
    mapper = {i: u for i, u in enumerate(graph.nodes())}
    if n_components > 2:
        x = embed_graph_mds(graph, n_components, weight)
        x = canonical_direction(x)
    elif n_components == 2:
        pos = nx.kamada_kawai_layout(graph, weight=weight)
        x = np.array([list(pos[i]) for i in pos])
        x = canonical_direction(x)
        x = x.reshape(-1, 2)
    elif n_components == 1:
        x = embed_graph_mds(graph, n_components, weight)
        x = x.reshape(-1, 1)
    x = scale_largest_dim(x)
    pos = {mapper[i]: vec.reshape(-1) for i, vec in enumerate(x)}
    return pos


def make_bipartite_graph(graphs, encoding_func, count_threshold=2):
    feature_dict, feature_counts_dict = get_feature_dict(
        graphs, encoding_func, return_counts=True)
    embeddin_graph = nx.Graph()
    n_graphs = len(graphs)
    for i, g in enumerate(graphs):
        embeddin_graph.add_node(i, label='graph', ref=i)
        codes, fragments = encoding_func(g)
        for c in codes:
            if feature_counts_dict[c] >= count_threshold:
                j = n_graphs + c
                if j not in embeddin_graph.nodes():
                    embeddin_graph.add_node(j, label='frag', ref=c)
                if embeddin_graph.has_edge(i, j) is False:
                    embeddin_graph.add_edge(i, j, label='-', weight=1)
                else:
                    embeddin_graph.edges[i, j]['weight'] += 1
    for u, v in embeddin_graph.edges():
        # TODO: rescale to 0-1 and use 1-val
        embeddin_graph.edges[u, v]['len'] = 1 / \
            embeddin_graph.edges[u, v]['weight']
    return embeddin_graph


def make_graph(graphs, encoding_func):
    return make_bipartite_graph(graphs, encoding_func)


def fit_feature_embedding(graphs, encoding_func, n_components=2):
    g = make_graph(graphs, encoding_func)
    pos = embed(g, n_components=n_components, weight='len')
    feature_embeddings = dict()
    for u in g.nodes():
        if g.nodes[u]['label'] == 'frag':
            code = g.nodes[u]['ref']
            coord = pos[u]
            feature_embeddings[code] = coord
    return feature_embeddings


def graph_embedding(
        graphs, feature_embeddings, encoding_func,
        importance_dict=None, intercept=None):
    x = []
    for i, g in enumerate(graphs):
        codes, fragments = encoding_func(g)
        if importance_dict:
            mtx = [feature_embeddings[code] * importance_dict[code]
                   for code in codes if code in feature_embeddings]
        else:
            mtx = [feature_embeddings[code]
                   for code in codes if code in feature_embeddings]
        mtx = np.array(mtx)
        # vec = mtx.mean(axis=0).reshape(-1)
        vec = mtx.sum(axis=0).reshape(-1)
        if intercept:
            vec = vec + intercept
        x.append(vec)
    x = np.vstack(x)
    return x


class Embedder(object):

    def __init__(self,
                 encoding_func,
                 importance_dict=None,
                 intercept=None,
                 n_components=2,
                 count_threshold=1):
        self.encoding_func = encoding_func
        self.n_components = n_components
        self.count_threshold = count_threshold
        self.importance_dict = importance_dict
        self.intercept = intercept

    def fit(self, graphs):
        self.feature_embeddings = fit_feature_embedding(
            graphs,
            encoding_func=self.encoding_func,
            n_components=self.n_components,
            count_threshold=self.count_threshold)
        return self

    def transform(self, graphs):
        x = graph_embedding(
            graphs,
            feature_embeddings=self.feature_embeddings,
            encoding_func=self.encoding_func,
            importance_dict=self.importance_dict,
            intercept=self.intercept)
        return x


def embed2D(graphs, decomposition_funcs, preprocessors=None, nbits=11):
    feature_size, bitmask = set_feature_size(nbits=nbits)
    encoding_func = make_encoder(
        decomposition_funcs, preprocessors=None, bitmask=bitmask, seed=1)
    emb = Embedder(encoding_func, n_components=2)
    return emb.fit(graphs).transform(graphs)


def plot2D(graphs, decomposition_funcs, preprocessors=None, nbits=11,
           colors=None, size=8, display_ids=False):
    x = embed2D(graphs, decomposition_funcs, preprocessors, nbits)
    a, b = x.T

    plt.figure(figsize=(size, size))
    if colors is None:
        plt.scatter(a, b, marker='o', edgecolors='k', alpha=.4)
    else:
        plt.scatter(a, b, c=colors, marker='o', edgecolors='k', alpha=.4)
    if display_ids:
        for i, row in enumerate(x):
            a, b = row
            plt.text(a, b, str(i))
    plt.grid()
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
