#!/usr/bin/env python
"""Provides scikit interface."""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm as normal_distribution
import numpy as np
from ego.vectorize import vectorize
import logging

logger = logging.getLogger()


class ExpectedImprovementEstimator(object):
    """ExpectedImprovementEstimator."""

    def __init__(self):
        """init."""
        kernel = 1.0 * Matern(length_scale=0.001, nu=1.5) + \
            1.0 * RationalQuadratic(length_scale=0.001,
                                    alpha=0.1) + WhiteKernel(noise_level=0.01)
        self.gp_estimator = GaussianProcessRegressor(
            n_restarts_optimizer=10, kernel=kernel)
        self.loss_optimum = 0

    def fit(self, x, y):
        """fit."""
        _num_attempts_ = 10
        self.loss_optimum = max(y)
        logger.debug('Current optimum: %.4f' % self.loss_optimum)
        for it in range(_num_attempts_):
            try:
                self.gp_estimator.fit(x, y)
            except Exception as e:
                logger.debug('Error: %s' % e)
            else:
                break
        loglike = self.gp_estimator.log_marginal_likelihood(
            self.gp_estimator.kernel_.theta)
        logger.debug("Posterior (kernel: %s)\n Log-Likelihood: %.3f (%.3f)" % (
            self.gp_estimator.kernel_, loglike, loglike / len(y)))

    def predict_mean_and_variance(self, x):
        """predict_mean_and_variance."""
        mu, sigma = self.gp_estimator.predict(x, return_std=True)
        return mu, sigma

    def predict(self, x):
        """predict."""
        mu, sigma = self.predict_mean_and_variance(x)
        with np.errstate(divide='ignore'):
            z = (mu - self.loss_optimum) / sigma
            a = (mu - self.loss_optimum) * normal_distribution.cdf(z)
            b = sigma * normal_distribution.pdf(z)
            exp_imp = a + b
            exp_imp[sigma == 0.0] = 0
        return exp_imp


class EmbeddedGraphExpectedImprovementEstimator(object):
    """GraphExpectedImprovementEstimator."""

    def __init__(self, graph_vector_embedder=None):
        """init."""
        self.exp_imp_estimator = ExpectedImprovementEstimator()
        self.graph_vector_embedder = graph_vector_embedder

    def fit(self, graphs, targets):
        """fit."""
        self.graph_vector_embedder.fit(graphs, targets)
        x = self.graph_vector_embedder.transform(graphs)
        self.exp_imp_estimator.fit(x, targets)
        return self

    def transform(self, graphs):
        """transform."""
        x = self.graph_vector_embedder.transform(graphs)
        return x

    def expected_improvement(self, x):
        """expected_improvement."""
        return self.exp_imp_estimator.predict(x)

    def predict(self, graphs):
        """predict."""
        x = self.graph_vector_embedder.transform(graphs)
        y = self.exp_imp_estimator.predict(x)
        return y


class GraphExpectedImprovementEstimator(object):
    """GraphExpectedImprovementEstimator."""

    def __init__(self, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1, eps=1, exploitation_vs_exploration=1):
        """init."""
        self.decomposition_funcs = decomposition_funcs
        self.preprocessors = preprocessors
        self.nbits = nbits
        self.seed = seed
        self.eps = eps
        self.exploitation_vs_exploration = exploitation_vs_exploration

    def _recondition(self, mtx, eps=1e-1):
        U, s, Vh = np.linalg.svd(mtx)
        s[s < eps] = eps
        mtx_p = U * np.mat(np.diag(s)) * Vh
        return mtx_p

    def fit(self, graphs, targets):
        """fit."""
        self.graphs = graphs[:]
        self.data_scores = np.mat(targets).reshape(-1, 1)
        self.loss_optimum = max(targets)
        self.data_mtx = vectorize(graphs, self.decomposition_funcs,
                                  self.preprocessors, self.nbits, self.seed)
        # compute inverse of Gram matrix
        gram_mtx = np.mat(self.data_mtx.dot(self.data_mtx.T).A)
        gram_mtx = self._recondition(gram_mtx, self.eps)
        self.gram_mtx_inv = gram_mtx.I
        return self

    def predict_mean_and_std(self, graphs):
        """predict_mean_and_std."""
        mus = []
        sigmas = []
        for graph in graphs:
            input_data_mtx = vectorize(
                [graph], self.decomposition_funcs,
                self.preprocessors, self.nbits, self.seed)
            input_gram_vec = np.mat(input_data_mtx.dot(self.data_mtx.T).A)
            mu = input_gram_vec * self.gram_mtx_inv * self.data_scores
            sigma2 = input_data_mtx.dot(
                input_data_mtx.T) - input_gram_vec * self.gram_mtx_inv * input_gram_vec.T
            mus.append(mu[0, 0])
            sigmas.append(np.sqrt(sigma2[0, 0]))
        mu, sigma = np.array(mus), np.array(sigmas)
        return mu, sigma

    def predict_mean(self, graphs):
        """predict_mean."""
        mu, sigma = self.predict_mean_and_std(graphs)
        return mu

    def predict_expected_improvement(self, graphs):
        """predict_expected_improvement."""
        mu, sigma = self.predict_mean_and_std(graphs)
        with np.errstate(divide='ignore'):
            z = (mu - self.loss_optimum) / sigma
            a = (mu - self.loss_optimum) * normal_distribution.cdf(z)
            b = sigma * normal_distribution.pdf(z)
            exp_imp = a * self.exploitation_vs_exploration + \
                (1 - self.exploitation_vs_exploration) * b
            exp_imp[sigma == 0.0] = 0
            exp_imp[exp_imp < 0] = 0
            exp_imp[np.isnan(exp_imp)] = 0
        return exp_imp
