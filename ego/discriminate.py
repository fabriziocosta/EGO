#!/usr/bin/env python
"""Provides scikit interface."""

from sklearn.linear_model import SGDClassifier
from ego.vectorize import set_feature_size
from ego.encode import make_encoder
from ego.vectorize import vectorize_graphs


class Discriminator(object):

    def __init__(self, decompose_func=None, preprocessor=None, nbits=14):
        feature_size, bitmask = set_feature_size(nbits=nbits)
        self.feature_size = feature_size
        self.bitmask = bitmask
        self.encoding_func = make_encoder(
            decompose_func,
            preprocessors=preprocessor,
            bitmask=self.bitmask,
            seed=1)
        self.estimator = SGDClassifier(penalty='elasticnet', tol=1e-3)

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
        preds = self.estimator.decision_function(x)
        return preds

    def predict(self, graphs):
        x = vectorize_graphs(
            graphs,
            encoding_func=self.encoding_func,
            feature_size=self.feature_size)
        preds = self.estimator.predict(x)
        return preds
