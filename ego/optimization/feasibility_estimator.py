#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
from ego.vectorize import vectorize
import logging

logger = logging.getLogger()


class FeasibilityEstimator(object):
    """FeasibilityEstimator."""

    def __init__(self, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1):
        """init."""
        self.decomposition_funcs = decomposition_funcs
        self.preprocessors = preprocessors
        self.nbits = nbits
        self.seed = seed

    def fit(self, graphs):
        """fit."""
        data_mtx = self.transform(graphs)
        all_feats = data_mtx.sum(axis=0)
        all_feats[all_feats != 0] = 1
        self.infeasibility_vec = np.logical_not(all_feats).astype(int)
        return self

    def transform(self, graphs):
        """transform."""
        data_mtx = vectorize(graphs, self.decomposition_funcs,
                             self.preprocessors, self.nbits, self.seed)
        return data_mtx

    def predict(self, graphs):
        """predict."""
        data_mtx = self.transform(graphs)
        infeasibilities = data_mtx.dot(self.infeasibility_vec.T).A.reshape(-1).astype(bool)
        feasibilities = np.logical_not(infeasibilities)
        return feasibilities

    def filter(self, graphs):
        """filter."""
        feasibilities = self.predict(graphs)
        return [g for g, f in zip(graphs, feasibilities) if f]
