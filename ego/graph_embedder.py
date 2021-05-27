#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import networkx as nx
from eden.graph import vectorize as eden_vectorize
from ego.vectorize import vectorize as ego_vectorize
from ego.node_vectorize import graph_node_vectorize as ego_node_vectorize
#from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS


from umap import UMAP
from toolz import curry

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

import logging
logger = logging.getLogger()


class EdenTransformer(object):

    def __init__(self, complexity=None, nbits=16):
        self.vectorize = curry(eden_vectorize)(complexity=complexity, nbits=nbits)

    def fit(self, graphs, y=None):
        return self

    def transform(self, graphs):
        return self.vectorize(graphs)

    def fit_transform(self, graphs, y=None):
        return self.fit(graphs, y).transform(graphs)


class EgoTransformer(object):

    def __init__(
            self,
            decomposition_funcs=None,
            preprocessors=None,
            nbits=16,
            seed=1):
        self.vectorize = curry(ego_vectorize)(
            decomposition_funcs=decomposition_funcs,
            preprocessors=preprocessors,
            nbits=nbits,
            seed=seed)

    def fit(self, graphs, y=None):
        return self

    def transform(self, graphs):
        return self.vectorize(graphs)

    def fit_transform(self, graphs, y=None):
        return self.fit(graphs, y).transform(graphs)

class EgoNodeTransformer(object):

    def __init__(
            self,
            decomposition_funcs=None,
            preprocessors=None,
            effective_radius=1,
            max_num_node_features=1,
            nbits=16):
        self.vectorize = curry(ego_node_vectorize)(
            decomposition_funcs=decomposition_funcs,
            preprocessors=preprocessors,
            effective_radius=effective_radius,
            max_num_node_features=max_num_node_features,
            nbits=nbits)

    def fit(self, graphs, y=None):
        return self

    def transform(self, graphs):
        return self.vectorize(graphs)

    def fit_transform(self, graphs, y=None):
        return self.fit(graphs, y).transform(graphs)


class SVDTransformer(object):

    def __init__(self, n_components=10):
        self.n_components = n_components
        if self.n_components is None:
            self.svd = TruncatedSVD(n_components=2)
        else:
            self.svd = TruncatedSVD(n_components=n_components)

    def fit(self, x, y=None):
        if self.n_components is None:
            self.n_components = min(x.shape) - 1
            self.svd = TruncatedSVD(n_components=self.n_components)
        self.svd.fit(x)
        return self

    def transform(self, x):
        return self.svd.transform(x)

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)


class AutoEncoder(object):

    def __init__(
            self,
            input_dim=1024,
            dim_enc_layer_1=256,
            dim_enc_layer_2=128,
            encoding_dim=64,
            dim_dec_layer_1=128,
            dim_dec_layer_2=256,
            batch_size=256,
            epochs=10000,
            patience=50,
            min_delta=0,
            log_dir=None,
            verbose=0):
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta

        self.log_dir = log_dir
        self.verbose = verbose

        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(dim_enc_layer_1, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(dim_enc_layer_2, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(self.encoding_dim, activation='sigmoid')(encoded)

        decoded = Dense(dim_dec_layer_1, activation='relu')(encoded)
        decoded = Dense(dim_dec_layer_2, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adadelta',
                                 loss='binary_crossentropy')

        # this model maps an input to its encoded representation
        self.encoder = Model(input_layer, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.encoding_dim,))
        # retrieve the last layer of the autoencoder model
        deco = self.autoencoder.layers[-3](encoded_input)
        deco = self.autoencoder.layers[-2](deco)
        deco = self.autoencoder.layers[-1](deco)
        # create the decoder model
        self.decoder = Model(encoded_input, deco)

    def fit(self, x_train, x_val=None):
        callbacks = []
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=self.patience, min_delta=self.min_delta))
        if self.log_dir is not None:
            callbacks.append(TensorBoard(log_dir=self.log_dir))
        if x_val is None:
            x_val = x_train
        self.autoencoder.fit(
            x_train, x_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(x_val, x_val),
            callbacks=callbacks,
            verbose=self.verbose)
        return self

    def transform(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        encods = self.encoder.predict(x)
        decods = self.decoder.predict(encods)
        return decods


class MinMaxNormalizeTransformer(object):

    def __init__(self):
        self.minmaxscaler = MinMaxScaler()
        self.normalizer = Normalizer()

    def fit(self, x, y=None):
        self.normalizer.fit(self.minmaxscaler.fit_transform(x))
        return self

    def transform(self, x):
        return self.normalizer.transform(self.minmaxscaler.transform(x))

    def fit_transform(self, x, y):
        return self.fit(x, y).transform(x)


class AutoEncoderTransformer(object):

    def __init__(self, autoencoder):
        self.autoencoder = autoencoder

    def fit(self, x, y=None):
        self.autoencoder.fit(x)
        return self

    def transform(self, x):
        return self.autoencoder.transform(x)

    def fit_transform(self, x, y):
        return self.fit(x, y).transform(x)


class UmapTransformer(object):

    def __init__(self, n_components=2, embed_n_neighbors=10, target_metric='categorical'):
        self.umap = UMAP(
            n_components=n_components,
            n_neighbors=embed_n_neighbors,
            target_metric=target_metric,
            transform_seed=1)
        self.rf = RandomForestRegressor(n_estimators=300)

    def fit(self, x, y=None):
        self.umap.fit(x, y)
        x_low = self.umap.transform(x)
        self.rf.fit(x, x_low)
        return self

    def transform(self, x):
        return self.rf.predict(x)

    def fit_transform(self, x, y):
        return self.fit(x, y).transform(x)


# class NcaTransformer(object):

#     def __init__(self, n_components=2):
#         self.nca = NeighborhoodComponentsAnalysis(n_components=n_components)

#     def fit(self, x, y=None):
#         self.nca.fit(x, y)
#         return self

#     def transform(self, x):
#         return self.mds.transform(x)

#     def fit_transform(self, x, y):
#         return self.fit(x, y).transform(x)


class RotoScaleTransformer(object):

    def __init__(self):
        self.rotation = self.xmin = self.xmax = self.xlen = None

    def fit(self, x, y=None):
        # rotation
        u, s, vh = np.linalg.svd(x, full_matrices=True)
        self.rotation = vh.T
        x_low_rot = np.dot(x, self.rotation)

        # scale
        self.xmin = np.min(x_low_rot, axis=0).reshape(-1)
        self.xmax = np.max(x_low_rot, axis=0).reshape(-1)
        self.xlen = np.absolute(self.xmax - self.xmin).max()
        return self

    def transform(self, x):
        x_low_rot = np.dot(x, self.rotation)
        x_low_rot_rescaled = (x_low_rot - self.xmin) / self.xlen
        return x_low_rot_rescaled

    def fit_transform(self, x, y):
        return self.fit(x, y).transform(x)


class ExtraTreesTransformer(object):

    def __init__(self, task='classification', n_estimators=1000):
        if task == 'classification':
            self.extratrees = ExtraTreesClassifier(n_estimators=n_estimators, random_state=42)
        else:
            self.extratrees = ExtraTreesRegressor(n_estimators=n_estimators, random_state=42)

    def fit(self, x, y=None):
        if y is None:
            y = np.random.randint(2, size=x.shape[0])
        self.extratrees.fit(x, y)
        return self

    def transform(self, x):
        xy, _ = self.extratrees.decision_path(x)
        return xy

    def fit_transform(self, x, y):
        return self.fit(x, y).transform(x)


# ----------------------------------------------------------------------------

def make_knn(X, n_neighbors):
    A = kneighbors_graph(X, n_neighbors=n_neighbors)
    g = nx.from_numpy_matrix(A.todense())
    return g


def make_iterated_mst(X, n_iter_mst):
    D = pairwise_distances(X)
    dt = [('weight', float)]
    C = np.array(D, dtype=dt)
    gd = nx.from_numpy_matrix(C)

    for i in range(n_iter_mst):
        gmst = nx.minimum_spanning_tree(gd, weight='weight')
        if i == 0:
            g = gmst
        else:
            g = nx.compose(g, gmst)
        gdd = nx.difference(gd, g)
        gd = gd.edge_subgraph(gdd.edges()).copy()
    return g


def make_perturbed_iterated_mst(X, n_iter_mst, n_iter_perturb):
    gg = make_iterated_mst(X, n_iter_mst)

    size = X.shape[0]
    for it in range(n_iter_perturb):
        idx = np.random.randint(0, size, size)
        Xp = X[idx]
        g = make_iterated_mst(Xp, n_iter_mst)
        mapping = {i: j for i, j in enumerate(idx)}
        h = nx.relabel_nodes(g, mapping)
        gg = nx.compose(gg, h)
    return gg


def make_knn_mst(X, n_neighbors, n_iter_mst, n_iter_perturb):
    if n_neighbors == 0:
        g_knn = nx.Graph()
    else:
        g_knn = make_knn(X, n_neighbors)
    if n_iter_mst == 0:
        g_mst = nx.Graph()
    else:
        g_mst = make_perturbed_iterated_mst(X, n_iter_mst, n_iter_perturb)
    g = nx.compose(g_knn, g_mst)
    return g


def euclidean_distance(x, z):
    return np.linalg.norm(x - z)


def annotate_graph(g, X, y):
    if y is not None:
        for u, t in zip(g.nodes(), y):
            g.nodes[u]['label'] = t
    else:
        for u in g.nodes():
            g.nodes[u]['label'] = '-'
    for u, v in g.edges():
        g.edges[u, v]['label'] = '-'
        g.edges[u, v]['weight'] = euclidean_distance(X[u], X[v])


def get_distances(g):
    n = g.number_of_nodes()
    distance_mtx = np.zeros((n, n))
    dist_dict = dict(nx.shortest_path_length(g, weight='weight'))
    for k in sorted(dist_dict):
        for j in sorted(dist_dict[k]):
            distance_mtx[k, j] = dist_dict[k][j]
    return distance_mtx


class KNearestNeighborMinimumSpannigTreeTransformer(object):

    def __init__(self, n_neighbors=5, n_iter_mst=2, n_iter_perturb=5, n_components=None, n_init=5, max_iter=10000, n_estimators=10000, add_original_features=False):
        self.n_neighbors = n_neighbors
        self.n_iter_mst = n_iter_mst
        self.n_iter_perturb = n_iter_perturb
        self.n_components = n_components
        self.add_original_features = add_original_features
        self.mds = MDS(n_components=n_components, n_init=n_init, max_iter=max_iter, dissimilarity='precomputed')
        self.rf = RandomForestRegressor(n_estimators=n_estimators)

    def _fit(self, x, y=None):
        if self.n_components is None:
            self.n_components = max(2, int(np.sqrt(x.shape[1])))
            self.mds.set_params(n_components=self.n_components)
        g = make_knn_mst(x, n_neighbors=self.n_neighbors, n_iter_mst=self.n_iter_mst, n_iter_perturb=self.n_iter_perturb)
        annotate_graph(g, x, y)
        distance_mtx = get_distances(g)
        x_low = self.mds.fit_transform(distance_mtx)
        if self.add_original_features:
            x_low = np.hstack([x_low, x])
        return x_low

    def fit(self, x, y=None):
        x_low = self._fit(x, y)
        self.rf.fit(x, x_low)
        return self

    def transform(self, x):
        return self.rf.predict(x)

    def fit_transform(self, x, y=None):
        x_low = self._fit(x, y)
        self.rf.fit(x, x_low)
        return x_low


#------------------------------------------------------------------------------


class Embedder(object):

    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, instances, y=None):
        """fit."""
        x = self.transformers[0].fit_transform(instances, y)
        for transformer in self.transformers[1:]:
            x = transformer.fit_transform(x, y)
        return self

    def transform(self, instances):
        """transform."""
        x = self.transformers[0].transform(instances)
        for transformer in self.transformers[1:]:
            x = transformer.transform(x)
        return x

    def fit_transform(self, instances, y=None):
        x = self.transformers[0].fit_transform(instances, y)
        for transformer in self.transformers[1:]:
            x = transformer.fit_transform(x, y)
        return x
