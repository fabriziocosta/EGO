#!/usr/bin/env python
"""Provides scikit interface."""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import Matern
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from scipy.stats import norm as normal_distribution
import numpy as np
from ego.vectorize import vectorize
import logging

logger = logging.getLogger()


class ScoreEstimator(object):
    """ScoreEstimator."""

    def __init__(self, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1):
        """init."""
        self.decomposition_funcs = decomposition_funcs
        self.preprocessors = preprocessors
        self.nbits = nbits
        self.seed = seed

    def fit(self, graphs, targets):
        """fit."""
        return self

    def transform(self, graphs):
        """transform."""
        assert False, 'Is not implemented.'
        return None

    def predict(self, graphs):
        """predict."""
        assert False, 'Is not implemented.'
        return None

    def predict_uncertainty(self, graphs):
        """predict_uncertainty."""
        assert False, 'Is not implemented.'
        return None

    def predict_gradient(self, graphs):
        """predict_gradient."""
        assert False, 'Is not implemented.'
        return None


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


class GraphUpperConfidenceBoundEstimator(ScoreEstimator):
    """GraphExpectedImprovementEstimator."""

    def __init__(self, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1, eps=1, exploration_vs_exploitation=0):
        """init."""
        self.decomposition_funcs = decomposition_funcs
        self.preprocessors = preprocessors
        self.nbits = nbits
        self.seed = seed
        self.eps = eps
        self.exploration_vs_exploitation = exploration_vs_exploitation

    def _recondition(self, mtx, eps=1e-1):
        U, s, Vh = np.linalg.svd(mtx)
        s[s < eps] = eps
        mtx_p = U * np.mat(np.diag(s)) * Vh
        return mtx_p

    def fit(self, graphs, targets):
        """fit."""
        self.graphs = graphs[:]
        self.data_scores = np.mat(targets).reshape(-1, 1)
        self.data_mtx = self.transform(graphs)
        # compute inverse of Gram matrix
        gram_mtx = np.mat(self.data_mtx.dot(self.data_mtx.T).A)
        gram_mtx = self._recondition(gram_mtx, self.eps)
        self.gram_mtx_inv = gram_mtx.I
        return self

    def transform(self, graphs):
        """transform."""
        data_mtx = vectorize(graphs, self.decomposition_funcs,
                             self.preprocessors, self.nbits, self.seed)
        return data_mtx

    def predict_mean_and_std(self, graphs):
        """predict_mean_and_std."""
        mus = []
        sigmas = []
        for graph in graphs:
            input_data_mtx = self.transform([graph])
            input_gram_vec = np.mat(input_data_mtx.dot(self.data_mtx.T).A)
            mu = input_gram_vec * self.gram_mtx_inv * self.data_scores
            sigma2 = input_data_mtx.dot(
                input_data_mtx.T) - input_gram_vec * self.gram_mtx_inv * input_gram_vec.T
            mus.append(mu[0, 0])
            sigmas.append(np.sqrt(sigma2[0, 0]))
        mu, sigma = np.array(mus), np.array(sigmas)
        return mu, sigma

    def predict(self, graphs):
        """predict_mean."""
        mu, sigma = self.predict_mean_and_std(graphs)
        outs = mu + self.exploration_vs_exploitation * sigma
        return outs

    def predict_uncertainty(self, graphs):
        """predict_uncertainty."""
        mu, sigma = self.predict_mean_and_std(graphs)
        return sigma


class GraphExpectedImprovementEstimator(ScoreEstimator):
    """GraphExpectedImprovementEstimator."""

    def __init__(self, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1, eps=1, exploration_vs_exploitation=0):
        """init."""
        self.decomposition_funcs = decomposition_funcs
        self.preprocessors = preprocessors
        self.nbits = nbits
        self.seed = seed
        self.eps = eps
        self.exploration_vs_exploitation = exploration_vs_exploitation

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
        self.data_mtx = self.transform(graphs)
        # compute inverse of Gram matrix
        gram_mtx = np.mat(self.data_mtx.dot(self.data_mtx.T).A)
        gram_mtx = self._recondition(gram_mtx, self.eps)
        self.gram_mtx_inv = gram_mtx.I
        return self

    def transform(self, graphs):
        """transform."""
        data_mtx = vectorize(graphs, self.decomposition_funcs,
                             self.preprocessors, self.nbits, self.seed)
        return data_mtx

    def predict_mean_and_std(self, graphs):
        """predict_mean_and_std."""
        mus = []
        sigmas = []
        for graph in graphs:
            input_data_mtx = self.transform([graph])
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

    def predict(self, graphs):
        """predict_expected_improvement."""
        mu, sigma = self.predict_mean_and_std(graphs)
        with np.errstate(divide='ignore'):
            z = (mu - self.loss_optimum) / sigma
            a = (mu - self.loss_optimum) * normal_distribution.cdf(z)
            b = sigma * normal_distribution.pdf(z)
            exp_imp = a * (1 - self.exploration_vs_exploitation) + \
                self.exploration_vs_exploitation * b
            exp_imp[sigma == 0.0] = 0
            exp_imp[exp_imp < 0] = 0
            exp_imp[np.isnan(exp_imp)] = 0
        return exp_imp

    def predict_uncertainty(self, graphs):
        """predict_uncertainty."""
        mu, sigma = self.predict_mean_and_std(graphs)
        return sigma


class GraphRandomForestScoreEstimator(object):
    """ScoreEstimator."""

    def __init__(self, n_estimators=100, exploration_vs_exploitation=0, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1):
        """init."""
        self.exploration_vs_exploitation = exploration_vs_exploitation
        self.decomposition_funcs = decomposition_funcs
        self.preprocessors = preprocessors
        self.nbits = nbits
        self.seed = seed
        self.estimator = RandomForestRegressor(n_estimators=n_estimators)

    def fit(self, graphs, targets):
        """fit."""
        self.data_mtx = self.transform(graphs)
        self.estimator = self.estimator.fit(self.data_mtx, targets)
        return self

    def transform(self, graphs):
        """transform."""
        data_mtx = vectorize(graphs, self.decomposition_funcs,
                             self.preprocessors, self.nbits, self.seed)
        return data_mtx

    def _predict(self, graphs):
        test_data_mtx = self.transform(graphs)
        preds = []
        for pred in self.estimator.estimators_:
            preds.append(pred.predict(test_data_mtx))
        preds = np.array(preds)
        return preds

    def predict(self, graphs):
        """predict."""
        if self.exploration_vs_exploitation == 0:
            outs = self.estimator.predict(self.transform(graphs))
        else:
            preds = self._predict(graphs)
            mu = np.mean(preds, axis=0)
            sigma = np.std(preds, axis=0)
            outs = mu + self.exploration_vs_exploitation * sigma
        return outs

    def predict_uncertainty(self, graphs):
        """predict_uncertainty."""
        test_data_mtx = self.transform(graphs)
        preds = []
        for pred in self.estimator.estimators_:
            preds.append(pred.predict(test_data_mtx))
        preds = np.array(preds)
        sigma = np.std(preds, axis=0)
        return sigma


class GraphLinearScoreEstimator(object):
    """ScoreEstimator."""

    def __init__(self, n_estimators=100, exploration_vs_exploitation=0, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1):
        """init."""
        self.exploration_vs_exploitation = exploration_vs_exploitation
        self.decomposition_funcs = decomposition_funcs
        self.preprocessors = preprocessors
        self.nbits = nbits
        self.seed = seed
        self.estimators = [SGDRegressor(penalty='elasticnet')
                           for _ in range(n_estimators)]

    def fit(self, graphs, targets):
        """fit."""
        y = np.array(targets)
        data_mtx = self.transform(graphs)
        size = data_mtx.shape[0]
        for i in range(len(self.estimators)):
            idx = np.random.randint(0, size, size)
            x_train = data_mtx[idx]
            y_train = y[idx]
            self.estimators[i] = self.estimators[i].fit(x_train, y_train)
        return self

    def transform(self, graphs):
        """transform."""
        data_mtx = vectorize(graphs, self.decomposition_funcs,
                             self.preprocessors, self.nbits, self.seed)
        return data_mtx

    def _predict(self, graphs):
        """predict."""
        data_mtx = self.transform(graphs)
        preds = [est.predict(data_mtx) for est in self.estimators]
        preds = np.array(preds)
        return preds

    def predict(self, graphs):
        """predict."""
        preds = self._predict(graphs)
        mu = np.mean(preds, axis=0)
        sigma = np.std(preds, axis=0)
        outs = mu + self.exploration_vs_exploitation * sigma
        return outs

    def predict_uncertainty(self, graphs):
        """predict_uncertainty."""
        preds = self._predict(graphs)
        sigma = np.std(preds, axis=0)
        return sigma

    def predict_gradient(self, graphs):
        """predict_gradient."""
        coefs = [est.coef_ for est in self.estimators]
        coefs = np.array(coefs)
        coef = np.mean(coefs, axis=0)
        return coef
