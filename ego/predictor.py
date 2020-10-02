#!/usr/bin/env python
"""Provides scikit interface."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.naive_bayes import BernoulliNB
from ego.vectorize import set_feature_size
from ego.encode import make_encoder
from ego.vectorize import vectorize_graphs
import numpy as np
from scipy.sparse import csr_matrix
from ego.vectorize import get_feature_dict
import math
from collections import Counter


class Classifier(object):

    def __init__(self, decompose_func=None, preprocessor=None, nbits=14, seed=1, n_estimators=10000):
        feature_size, bitmask = set_feature_size(nbits=nbits)
        self.feature_size = feature_size
        self.bitmask = bitmask
        self.encoding_func = make_encoder(
            decompose_func,
            preprocessors=preprocessor,
            bitmask=self.bitmask,
            seed=seed)
        self.estimator = ExtraTreesClassifier(
            n_estimators=n_estimators, random_state=seed)

    def fit(self, graphs, targets):
        x = vectorize_graphs(
            graphs,
            encoding_func=self.encoding_func,
            feature_size=self.feature_size)
        self.estimator.fit(x, targets)
        return self

    def decision_function(self, graphs):
        x = vectorize_graphs(
            graphs,
            encoding_func=self.encoding_func,
            feature_size=self.feature_size)
        preds = self.estimator.predict_proba(x)[:, 1].reshape(-1)
        return preds

    def predict(self, graphs):
        x = vectorize_graphs(
            graphs,
            encoding_func=self.encoding_func,
            feature_size=self.feature_size)
        preds = self.estimator.predict(x)
        return preds


class Regressor(object):

    def __init__(self, decompose_func=None, preprocessor=None, nbits=14, n_estimators=10000, seed=1):
        feature_size, bitmask = set_feature_size(nbits=nbits)
        self.feature_size = feature_size
        self.bitmask = bitmask
        self.encoding_func = make_encoder(
            decompose_func,
            preprocessors=preprocessor,
            bitmask=self.bitmask,
            seed=seed)
        self.estimator = ExtraTreesRegressor(
            n_estimators=n_estimators, random_state=seed)

    def fit(self, graphs, targets):
        x = vectorize_graphs(
            graphs,
            encoding_func=self.encoding_func,
            feature_size=self.feature_size)
        self.estimator.fit(x, targets)
        return self

    def predict(self, graphs):
        x = vectorize_graphs(
            graphs,
            encoding_func=self.encoding_func,
            feature_size=self.feature_size)
        preds = self.estimator.predict(x)
        return preds


def right_stochastic(mtx):
    d = np.asarray(mtx.sum(axis=1)).reshape(-1)
    diag_mtx = np.mat(np.diag(d))
    right_stochastic_mtx = np.dot(diag_mtx.I, mtx)
    return right_stochastic_mtx


class SimpleClassifier(object):

    def __init__(self, decompose_func=None, preprocessor=None, nbits=15, seed=1):
        feature_size, bitmask = set_feature_size(nbits=nbits)
        self.feature_size = feature_size
        self.bitmask = bitmask
        self.encoding_func = make_encoder(
            decompose_func,
            preprocessors=preprocessor,
            bitmask=self.bitmask,
            seed=seed)

    def fit(self, graphs, targets):
        data_mtx = vectorize_graphs(graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        # binarize
        data_mtx.data = np.where(data_mtx.data > 0, 1, 0)
        target_mtx = np.array([(0, 1) if t == 1 else (1, 0) for t in targets])
        self.target_bias = right_stochastic(target_mtx.sum(axis=0).reshape(1, -1))
        target_mtx = csr_matrix(target_mtx)
        self.classifier_mtx = target_mtx.T.dot(data_mtx)
        return self

    def decision_function(self, graphs):
        # return probability associated to largest target type
        data_mtx = vectorize_graphs(graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        prediction_mtx = data_mtx.dot(self.classifier_mtx.T).todense()
        preds = right_stochastic(prediction_mtx)
        # incorporate training set class bias
        preds = right_stochastic(preds * np.diag(np.asarray(self.target_bias).reshape(-1)))
        # assuming binary classification and column 1 to represent positives
        preds = preds[:, 1].A.reshape(-1)
        return preds

    def predict(self, graphs):
        preds = self.decision_function(graphs)
        targets = np.array([1 if p > .5 else 0 for p in preds])
        return targets


class NBClassifier(object):

    def __init__(self, decompose_func=None, preprocessor=None, nbits=15, seed=1):
        self.decompose_func = decompose_func
        self.nbits = nbits
        feature_size, bitmask = set_feature_size(nbits=nbits)
        self.feature_size = feature_size
        self.bitmask = bitmask
        self.encoding_func = make_encoder(
            decompose_func,
            preprocessors=preprocessor,
            bitmask=self.bitmask,
            seed=seed)
        self.classifier = BernoulliNB(
            alpha=0.1, binarize=None, fit_prior=True, class_prior=None)

    def fit(self, graphs, targets):
        data_mtx = vectorize_graphs(graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        # binarize
        data_mtx.data = np.where(data_mtx.data > 0, 1, 0)
        self.classifier.fit(data_mtx, targets)
        return self

    def decision_function(self, graphs):
        # return probability associated to largest target type
        data_mtx = vectorize_graphs(graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        # binarize
        data_mtx.data = np.where(data_mtx.data > 0, 1, 0)
        preds = self.classifier.predict_proba(data_mtx)
        # assuming binary classification and column 1 to represent positives
        preds = preds[:, 1].reshape(-1)
        return preds

    def predict(self, graphs):
        data_mtx = vectorize_graphs(graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        # binarize
        data_mtx.data = np.where(data_mtx.data > 0, 1, 0)
        preds = self.classifier.predict(data_mtx)
        return preds

    def explain(self, graphs, top_k):
        feature_dict, feature_counts = get_feature_dict(
            graphs,
            decomposition_funcs=self.decompose_func,
            nbits=self.nbits,
            return_counts=True)
        # compute log-odds
        scores = self.classifier.feature_log_prob_[1, :] / self.classifier.feature_log_prob_[0, :]
        ranked_pos_features = np.argsort(-scores)
        # signature-counts
        stats = [(feature_dict[id].graph['signature'], feature_counts[id]) for id in feature_dict]
        # aggregate counts according to same signature
        sig_dict = dict()
        for sig, c in stats:
            if sig in sig_dict:
                sig_dict[sig] += c
            else:
                sig_dict[sig] = c
        # take logs
        for id in sig_dict:
            sig_dict[id] = math.log(sig_dict[id])
        # select top_k
        feature_graphs = [feature_dict[fid] for fid in ranked_pos_features[:top_k]]
        c = Counter([g.graph['signature'] for g in feature_graphs])
        cnt = dict([(id, c[id] / sig_dict[id]) for id in c])
        tot = sum(cnt[id] for id in cnt)
        res = [(cnt[id] / tot, cnt[id], id) for id in sorted(cnt.keys(), key=lambda id:cnt[id], reverse=True)]
        return res


class LinearClassifier(object):

    def __init__(self, decompose_func=None, preprocessor=None, nbits=15, seed=1):
        self.decompose_func = decompose_func
        self.nbits = nbits
        feature_size, bitmask = set_feature_size(nbits=nbits)
        self.feature_size = feature_size
        self.bitmask = bitmask
        self.encoding_func = make_encoder(
            decompose_func,
            preprocessors=preprocessor,
            bitmask=self.bitmask,
            seed=seed)
        self.classifier = SGDClassifier(penalty='elasticnet')

    def fit(self, graphs, targets):
        data_mtx = vectorize_graphs(graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        # binarize
        data_mtx.data = np.where(data_mtx.data > 0, 1, 0)
        self.classifier.fit(data_mtx, targets)
        return self

    def decision_function(self, graphs):
        # return probability associated to largest target type
        data_mtx = vectorize_graphs(graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        # binarize
        data_mtx.data = np.where(data_mtx.data > 0, 1, 0)
        preds = self.classifier.decision_function(data_mtx)
        return preds

    def predict(self, graphs):
        data_mtx = vectorize_graphs(graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        # binarize
        data_mtx.data = np.where(data_mtx.data > 0, 1, 0)
        preds = self.classifier.predict(data_mtx)
        return preds


class LinearRegressor(object):

    def __init__(self, decompose_func=None, preprocessor=None, nbits=15, seed=1):
        self.decompose_func = decompose_func
        self.nbits = nbits
        feature_size, bitmask = set_feature_size(nbits=nbits)
        self.feature_size = feature_size
        self.bitmask = bitmask
        self.encoding_func = make_encoder(
            decompose_func,
            preprocessors=preprocessor,
            bitmask=self.bitmask,
            seed=seed)
        self.classifier = SGDRegressor(penalty='elasticnet')

    def fit(self, graphs, targets):
        data_mtx = vectorize_graphs(graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        # binarize
        data_mtx.data = np.where(data_mtx.data > 0, 1, 0)
        self.classifier.fit(data_mtx, targets)
        return self

    def decision_function(self, graphs):
        # return probability associated to largest target type
        data_mtx = vectorize_graphs(graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        # binarize
        data_mtx.data = np.where(data_mtx.data > 0, 1, 0)
        preds = self.classifier.decision_function(data_mtx)
        return preds

    def predict(self, graphs):
        data_mtx = vectorize_graphs(graphs, encoding_func=self.encoding_func, feature_size=self.feature_size)
        # binarize
        data_mtx.data = np.where(data_mtx.data > 0, 1, 0)
        preds = self.classifier.predict(data_mtx)
        return preds
