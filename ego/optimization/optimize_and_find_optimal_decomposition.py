#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
from ego.optimization.hyper_setup import hyperopt
from ego.decompose import do_decompose
from ego.decomposition.paired_neighborhoods import decompose_neighborhood
from ego.decomposition.cycle_basis import decompose_cycles, decompose_cycles_and_non_cycles
from ego.decomposition.join import decompose_edge_join, decompose_node_join
from ego.optimization.optimize import optimizer_setup, optimize, termination_condition, sample_with_expected_improvement
import logging

logger = logging.getLogger()


def find_optimal_decomposition(all_graphs, all_scores, order=2):

    # make pos neg from all_scores and all_graphs
    ids = np.argsort(all_scores)
    n = len(all_graphs)
    n_neg = int(1 * n / 2)
    n_pos = n - n_neg
    pos_graphs = [all_graphs[ids[-i]] for i in range(n_pos)]
    neg_graphs = [all_graphs[ids[i]] for i in range(n_neg)]

    # order = 2             # max n nodes in func tree
    data_size = 50         # max num pos/neg graphs
    memory = dict()        # all viable trees stored by order (i.e. num nodes)
    history = dict()       # all trees stored by hash
    auc_threshold = .45    # AUC ROC > auc_threshold
    cmpx_threshold = 15    # complexity < cmpx_threshold
    n_max = 15             # n of trees that are retained from the Pareto shells ordering: these are the top survivors at each iteration
    # n of trees that are materialized to be tried for performance: these will
    # be evaluated on data
    n_max_sel = 100
    n_max_sample = 200
    # max num sec for evaluating performance of a single decomposition
    # hypothesis
    timeout = 60 * 5
    max_n_hours = 20
    max_runtime = 3600 * max_n_hours
    test_frac = .50
    n_iter = 3

    opt_decompose_func, opt_decompose_func_str = hyperopt(
        pos_graphs, neg_graphs, data_size,
        memory, history, auc_threshold, cmpx_threshold,
        n_max, n_max_sel, n_max_sample, order,
        timeout, max_n_hours, max_runtime, test_frac, n_iter, display=True)
    logger.debug('Optimal decomposition: %s' % opt_decompose_func_str)
    return opt_decompose_func


def optimize_and_find_optimal_decomposition(init_graphs, domain_graphs, target_graphs, oracle_func, n_iter=10, max_decomposition_order=3, make_monitor=None):
    eights = do_decompose(
        decompose_cycles, composition_function=decompose_edge_join)
    nieghbs = do_decompose(decompose_neighborhood(
        radius=1), decompose_neighborhood(radius=2), decompose_neighborhood(radius=3))
    lollipops = do_decompose(
        decompose_cycles_and_non_cycles, compose_function=decompose_node_join)
    decomposition = do_decompose(decompose_cycles, eights, nieghbs, lollipops)

    graphs = init_graphs
    for it in range(n_iter):
        logger.info('-' * 100 + '\n iteration:%d/%d' % (it + 1, n_iter))
        # optimize starting from initial graphs
        n, ei, m = optimizer_setup(
            decomposition, domain_graphs, target_graphs, oracle_func,
            grammar_conservativeness=2, n_neighbors=3, exploitation_vs_exploration=1, make_monitor=make_monitor)
        graphs = optimize(graphs, oracle_func, n_iter=250, sample_size=5, max_pool_size=10, k_steps=1,
                          neighborhood_estimator=n, graph_expected_improvement_estimator=ei, monitor=m)

        # sample graphs generated during optimization
        # and learn optimal decomposition from them
        sample_graphs = sample_with_expected_improvement(graphs, 50, ei)
        sample_scores = [oracle_func(g) for g in sample_graphs]
        if termination_condition(sample_scores):
            break
        decomposition = find_optimal_decomposition(
            sample_graphs, sample_scores, order=max_decomposition_order)
    return decomposition, graphs
