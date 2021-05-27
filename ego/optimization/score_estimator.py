#!/usr/bin/env python
"""Provides scikit interface."""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import Matern
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import TruncatedSVD
from scipy.stats import norm as normal_distribution
from scipy.stats import rankdata
import numpy as np
from ego.vectorize import vectorize
from ego.utils.parallel_utils import simple_parallel_map
import logging
from toolz import curry

logger = logging.getLogger()


def inv_sigmoid(x, sigma=1e-2):
    """inv_sigmoid."""
    return 2 * (1 - (1 / (1 + np.exp(-x / sigma))))


@curry
def ensemble_score_estimator_fit(id, estimators=None, graphs=None, targets=None, unlabeled_graphs=None):
    """ensemble_score_estimator_fit."""
    out = estimators[id].fit(graphs, targets, unlabeled_graphs)
    return (id, out)


@curry
def ensemble_score_estimator_predict(id, estimators=None, graphs=None):
    """ensemble_score_estimator_predict."""
    out = estimators[id].predict(graphs)
    return (id, out)


class EnsembleScoreEstimator(object):
    """EnsembleScoreEstimator."""

    def __init__(self, estimators, execute_concurrently=False):
        """init."""
        self.estimators = estimators
        self.exploration_vs_exploitation = 0
        self.weights = None
        self.execute_concurrently = execute_concurrently

    def set_exploration_vs_exploitation(self, exploration_vs_exploitation):
        """set_exploration_vs_exploitation."""
        self.exploration_vs_exploitation = exploration_vs_exploitation
        for estimator in self.estimators:
            estimator.exploration_vs_exploitation = exploration_vs_exploitation

    def fit(self, graphs, targets, unlabeled_graphs=None):
        """fit."""
        if self.execute_concurrently is False:
            self.estimators = [estimator.fit(graphs, targets) for estimator in self.estimators]
        else:
            _ensemble_score_estimator_fit_ = ensemble_score_estimator_fit(estimators=self.estimators[:], graphs=graphs[:], targets=targets[:], unlabeled_graphs=unlabeled_graphs[:])
            results = simple_parallel_map(_ensemble_score_estimator_fit_, range(len(self.estimators)))
            self.estimators = [estimator for id, estimator in sorted(results, key=lambda x: x[0])]
            print('e--%s' % (self.estimators))
        return self

    def _predict(self, graphs):
        preds = []
        if self.execute_concurrently is False:
            preds = [estimator.predict(graphs) for estimator in self.estimators]
        else:
            _ensemble_score_estimator_predict_ = ensemble_score_estimator_predict(estimators=self.estimators[:], graphs=graphs[:])
            results = simple_parallel_map(_ensemble_score_estimator_predict_, range(len(self.estimators)))
            preds = [pred for id, pred in sorted(results, key=lambda x: x[0])]
            print('p--%s' % (preds))
        return preds

    def predict(self, graphs):
        """predict."""
        preds = self._predict(graphs)
        if self.weights is None:
            n = len(self.estimators)
            self.weights = np.array([1 / float(n)] * n)
        preds = np.sum((np.array(preds).T * np.array(self.weights).T).T, axis=0).reshape(-1)
        return preds

    def predict_uncertainty(self, graphs):
        """predict_uncertainty."""
        preds = []
        for estimator in self.estimators:
            preds.append(estimator.predict_uncertainty(graphs))
        if self.weights is None:
            n = len(self.estimators)
            self.weights = np.array([1 / float(n)] * n)
        preds = np.sum((np.array(preds).T * np.array(self.weights).T).T, axis=0).reshape(-1)
        return preds

    def acquisition_score(self, graphs):
        """acquisition_score."""
        preds = []
        for estimator in self.estimators:
            preds.append(estimator.acquisition_score(graphs))
        if self.weights is None:
            n = len(self.estimators)
            self.weights = np.array([1 / float(n)] * n)
        preds = np.sum((np.array(preds).T * np.array(self.weights).T).T, axis=0).reshape(-1)
        return preds

    def error(self, graphs, targets):
        """error."""
        errors = []
        for estimator in self.estimators:
            errors.append(np.mean(estimator.error(graphs, targets), axis=0))
        return errors

    def estimate_weights(self, graphs, targets):
        """estimate_weights."""
        errors = np.array(self.error(graphs, targets))
        w = rankdata(errors, method='ordinal').astype(float)
        w = 1 / w
        weights = w / np.sum(w)
        return weights, errors


class ScoreEstimatorInterface(object):
    """ScoreEstimatorInterface."""

    def __init__(self, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1, exploration_vs_exploitation=0):
        """init."""
        self.decomposition_funcs = decomposition_funcs
        self.preprocessors = preprocessors
        self.nbits = nbits
        self.seed = seed
        self.exploration_vs_exploitation = exploration_vs_exploitation

    def fit(self, graphs, targets, unlabeled_graphs=None):
        """fit."""
        return self

    def error(self, graphs, targets):
        """error."""
        assert False, 'Is not implemented.'
        return None

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

    def acquisition_score(self, graphs):
        """acquisition_score."""
        assert False, 'Is not implemented.'
        return None


class ScoreEstimator(ScoreEstimatorInterface):
    """ScoreEstimator."""

    def __init__(self, decomposition_funcs=None, preprocessors=None, nbits=14, seed=1, exploration_vs_exploitation=0, n_components=5):
        """init."""
        self.decomposition_funcs = decomposition_funcs
        self.preprocessors = preprocessors
        self.nbits = nbits
        self.seed = seed
        self.exploration_vs_exploitation = exploration_vs_exploitation
        self.estimators = []
        #self.embedder = TruncatedSVD(n_components=n_components)

    def _transform(self, graphs):
        data_mtx = vectorize(graphs, self.decomposition_funcs,
                             self.preprocessors, self.nbits, self.seed)
        return data_mtx

    def transform(self, graphs):
        """transform."""
        data_mtx = self._transform(graphs)
        #data_mtx = self.embedder.transform(graphs)
        return data_mtx
        
    def fit(self, graphs, targets, unlabeled_graphs=None):
        """fit."""
        y = np.array(targets)
        data_mtx = self.transform(graphs)
        #data_mtx = self.embedder.fit_transform(data_mtx)
        size = data_mtx.shape[0]
        n_estimators = len(self.estimators)
        n_splits = 5
        n_repeats = n_estimators // n_splits
        ids = [tr for tr, ts in RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=2652124).split(range(size))]
        for i, idx in zip(range(n_estimators), ids):
            # idx = np.random.randint(0, size, size)
            x_train = data_mtx[idx]
            y_train = y[idx]
            self.estimators[i] = self.estimators[i].fit(x_train, y_train)
        return self

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
        return mu

    def acquisition_score(self, graphs):
        """acquisition_score."""
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

    def error(self, graphs, targets):
        """error."""
        preds = self._predict(graphs)
        mu = np.mean(preds, axis=0)
        sigma = np.std(preds, axis=0)
        err = np.power(mu - targets, 2) + np.power(sigma, 2)
        err + np.mean(err)
        return err


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
        return mu

    def predict_uncertainty(self, graphs):
        """predict_uncertainty."""
        mu, sigma = self.predict_mean_and_std(graphs)
        return sigma

    def acquisition_score(self, graphs):
        """acquisition_score."""
        mu, sigma = self.predict_mean_and_std(graphs)
        outs = mu + self.exploration_vs_exploitation * sigma
        return outs

    def error(self, graphs, targets):
        """error."""
        mu, sigma = self.predict_mean_and_std(graphs)
        err = np.power(mu - targets, 2) + np.power(sigma, 2)
        err + np.mean(err)
        return err


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
        """predict."""
        mu, sigma = self.predict_mean_and_std(graphs)
        return mu

    def acquisition_score(self, graphs):
        """acquisition_score."""
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

    def error(self, graphs, targets):
        """error."""
        mu, sigma = self.predict_mean_and_std(graphs)
        err = np.power(mu - targets, 2) + np.power(sigma, 2)
        err + np.mean(err)
        return err


class GraphRandomForestScoreEstimator(ScoreEstimator):
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

    def _predict(self, graphs):
        test_data_mtx = self.transform(graphs)
        preds = []
        for pred in self.estimator.estimators_:
            preds.append(pred.predict(test_data_mtx))
        preds = np.array(preds)
        return preds

    def predict(self, graphs):
        """predict."""
        outs = self.estimator.predict(self.transform(graphs))
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


class GraphLinearScoreEstimator(ScoreEstimator):
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

    def predict_gradient(self, graphs):
        """predict_gradient."""
        coefs = [est.coef_ for est in self.estimators]
        coefs = np.array(coefs)
        coef = np.mean(coefs, axis=0)
        return coef


class GraphNeuralNetworkScoreEstimator(ScoreEstimator):
    """GraphNeuralNetworkScoreEstimator."""

    def __init__(self,
                 hidden_layer_sizes=[100, 50],
                 alpha=0.0001,
                 n_estimators=100,
                 exploration_vs_exploitation=0,
                 decomposition_funcs=None,
                 preprocessors=None,
                 nbits=14,
                 seed=1):
        """init."""
        self.exploration_vs_exploitation = exploration_vs_exploitation
        self.decomposition_funcs = decomposition_funcs
        self.preprocessors = preprocessors
        self.nbits = nbits
        self.seed = seed
        self.estimators = [MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha) for _ in range(n_estimators)]

    def predict_gradient(self, graphs):
        """predict_gradient."""
        coefs = [est.coef_ for est in self.estimators]
        coefs = np.array(coefs)
        coef = np.mean(coefs, axis=0)
        return coef


class GraphNearestNeighborScoreEstimator(ScoreEstimator):
    """GraphNearestNeighborScoreEstimator."""

    def __init__(self,
                 n_neighbors=5,
                 n_estimators=100,
                 exploration_vs_exploitation=0,
                 decomposition_funcs=None,
                 preprocessors=None,
                 nbits=14,
                 seed=1):
        """init."""
        self.exploration_vs_exploitation = exploration_vs_exploitation
        self.decomposition_funcs = decomposition_funcs
        self.preprocessors = preprocessors
        self.nbits = nbits
        self.seed = seed
        self.estimators = [KNeighborsRegressor(n_neighbors=n_neighbors) for _ in range(n_estimators)]
