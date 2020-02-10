#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
from ego.optimization.neighborhood_graph_grammar import NeighborhoodPartImportanceGraphGrammar
from ego.optimization.expected_improvement_estimator import GraphExpectedImprovementEstimator
from ego.decomposition.paired_neighborhoods import decompose_neighborhood
from ego.vectorize import hash_graph
import logging

logger = logging.getLogger()


def remove_duplicates_in_set(graphs_to_filter, graph_archive):
    df = decompose_neighborhood(radius=2)
    val_set = set([hash_graph(g, decomposition_funcs=df) for g in graph_archive])
    selected_graphs = [g for g in graphs_to_filter if hash_graph(g, decomposition_funcs=df) not in val_set]
    return selected_graphs


def biased_sample(graphs, scores, sample_size):
    p = np.array(scores)
    # replace negative or zero probabilities with smallest positive prob
    p[p < 0] = 0
    min_p = np.min(p[p > 0])
    p[p == 0] = min_p
    p = p / p.sum()
    sample_ids = np.random.choice(len(graphs), size=sample_size - 1, replace=False, p=p)
    sample_graphs = [graphs[sample_id] for sample_id in sample_ids]
    return sample_graphs


def sample(graphs, scores, sample_size, greedy_frac=0.5):
    if sample_size > len(scores):
        sample_size = len(scores)

    # select with greedy strategy
    n_greedy_selection = int(sample_size * greedy_frac)
    sorted_ids = np.argsort(-np.array(scores))
    greedy_selected_ids = sorted_ids[:n_greedy_selection]
    greedy_selected_graphs = [graphs[greedy_selected_id] for greedy_selected_id in greedy_selected_ids]

    # select with policy: biased sample
    unselected_ids = sorted_ids[n_greedy_selection:]
    unselected_graphs = [graphs[unselected_id] for unselected_id in unselected_ids]
    unselected_scores = [scores[unselected_id] for unselected_id in unselected_ids]
    policy_selected_graphs = biased_sample(unselected_graphs, unselected_scores, sample_size - len(greedy_selected_graphs))

    selected_graphs = greedy_selected_graphs + policy_selected_graphs
    return selected_graphs


def sample_with_expected_improvement(graphs, sample_size, graph_expected_improvement_estimator):
    if graph_expected_improvement_estimator.exploitation_vs_exploration == 1:
        scores = graph_expected_improvement_estimator.predict_mean(graphs)
    else:
        scores = graph_expected_improvement_estimator.predict_expected_improvement(
            graphs)
    return sample(graphs, scores, sample_size, greedy_frac=0.5)


def perturb(graphs, neighborhood_estimator, k_steps=2):
    # generate a fixed num of neighbors for each graph in input
    neighbor_graphs = []
    for g in graphs:
        neighbor_graphs.extend(neighborhood_estimator.neighbors(g))
    # iterate the neighbour extraction k_steps times
    # this allows to consider moves that do not lead immediately to an improved solution
    # but that could lead to it in multiple steps
    for k in range(k_steps - 1):
        next_neighbor_graphs = []
        for g in neighbor_graphs:
            next_neighbor_graphs.extend(neighborhood_estimator.neighbors(g))
        neighbor_graphs = next_neighbor_graphs[:]
    return neighbor_graphs


def termination_condition(scores):
    return max(scores) > .99


def optimize(init_graphs, oracle_func, n_iter=100, sample_size=5, max_pool_size=25, k_steps=1,
             neighborhood_estimator=None, graph_expected_improvement_estimator=None, monitor=None):
    graphs = init_graphs[:]
    scores = [oracle_func(g) for g in graphs]
    proposed_graphs = []
    for i in range(n_iter):
        # update with oracle
        graphs += proposed_graphs
        scores += [oracle_func(g) for g in proposed_graphs]

        if termination_condition(scores):
            break

        # update part_importance_estimator
        neighborhood_estimator.fit_part_importance_estimator(graphs, scores)

        # update expected_improvement_estimator
        graph_expected_improvement_estimator.fit(graphs, scores)

        # sample neighborhood with EI
        sample_graphs = sample(graphs, scores, max_pool_size)
        neighbor_graphs = perturb(sample_graphs, neighborhood_estimator, k_steps=k_steps)
        neighbor_graphs = remove_duplicates_in_set(neighbor_graphs, graphs)
        proposed_graphs = sample_with_expected_improvement(neighbor_graphs, sample_size, graph_expected_improvement_estimator)

        if monitor:
            monitor(i, proposed_graphs, graphs, graph_expected_improvement_estimator,
                    neighborhood_estimator.part_importance_estimator)
    return graphs


def optimizer_setup(decomposition, domain_graphs, target_graphs, oracle_func,
                    grammar_conservativeness=2, n_neighbors=3, exploitation_vs_exploration=1, make_monitor=None):
    # performance monitor
    monitor = make_monitor(target_graphs, oracle_func, show_step=2)

    # neighborhood generator
    neighborhood_estimator = NeighborhoodPartImportanceGraphGrammar(
        decomposition_function=decomposition, context=2, count=grammar_conservativeness, frac_nodes_to_select=.5, n_neighbors=n_neighbors)
    neighborhood_estimator.fit_grammar(domain_graphs)
    logger.info(neighborhood_estimator)

    graph_expected_improvement_estimator = GraphExpectedImprovementEstimator(
        decomposition_funcs=decomposition, exploitation_vs_exploration=exploitation_vs_exploration)

    return neighborhood_estimator, graph_expected_improvement_estimator, monitor
