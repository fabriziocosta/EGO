#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import scipy as sp
import random

from ego.optimization.part_importance_estimator import PartImportanceEstimator

from ego.optimization.neighborhood_graph_grammar import NeighborhoodAdaptiveGraphGrammar
from ego.optimization.neighborhood_graph_grammar import NeighborhoodPartImportanceGraphGrammar

from ego.optimization.neighborhood_edge_swap import NeighborhoodEdgeSwap
from ego.optimization.neighborhood_edge_label_swap import NeighborhoodEdgeLabelSwap
from ego.optimization.neighborhood_edge_label_mutation import NeighborhoodEdgeLabelMutation
from ego.optimization.neighborhood_edge_move import NeighborhoodEdgeMove
from ego.optimization.neighborhood_edge_remove import NeighborhoodEdgeRemove
from ego.optimization.neighborhood_edge_add import NeighborhoodEdgeAdd
from ego.optimization.neighborhood_edge_expand import NeighborhoodEdgeExpand
from ego.optimization.neighborhood_edge_contract import NeighborhoodEdgeContract

from ego.optimization.neighborhood_node_label_swap import NeighborhoodNodeLabelSwap
from ego.optimization.neighborhood_node_label_mutation import NeighborhoodNodeLabelMutation
from ego.optimization.neighborhood_node_remove import NeighborhoodNodeRemove
from ego.optimization.neighborhood_node_add import NeighborhoodNodeAdd
from ego.optimization.neighborhood_node_smooth import NeighborhoodNodeSmooth

from ego.optimization.neighborhood_edge_crossover import NeighborhoodEdgeCrossover
from ego.optimization.neighborhood_cycle_crossover import NeighborhoodCycleCrossover
from ego.optimization.neighborhood_double_edge_crossover import NeighborhoodDoubleEdgeCrossover

from ego.optimization.score_estimator import GraphUpperConfidenceBoundEstimator
from ego.optimization.score_estimator import GraphRandomForestScoreEstimator
from ego.optimization.score_estimator import GraphLinearScoreEstimator
from ego.optimization.score_estimator import GraphExpectedImprovementEstimator
from ego.optimization.score_estimator import GraphNeuralNetworkScoreEstimator
from ego.optimization.score_estimator import GraphNearestNeighborScoreEstimator
from ego.optimization.score_estimator import EnsembleScoreEstimator

from ego.optimization.feasibility_estimator import FeasibilityEstimator
from ego.decomposition.paired_neighborhoods import decompose_neighborhood
from ego.vectorize import hash_graph
from ego.utils.parallel_utils import simple_parallel_map
# from ego.optimization.score_estimator import GraphLinearClassifier
import time
from toolz import curry
import logging

logger = logging.getLogger()


def remove_duplicates(graphs):
    """remove_duplicates."""
    df = decompose_neighborhood(radius=2)
    selected_graphs_dict = {hash_graph(
        g, decomposition_funcs=df): g for g in graphs}
    return list(selected_graphs_dict.values())


def remove_duplicates_in_set(graphs_to_filter, graph_archive):
    """remove_duplicates_in_set."""
    df = decompose_neighborhood(radius=2)
    val_set = set([hash_graph(g, decomposition_funcs=df)
                   for g in graph_archive])
    selected_graphs = [g for g in graphs_to_filter if hash_graph(
        g, decomposition_funcs=df) not in val_set]
    return selected_graphs


def biased_sample(graphs, scores, sample_size):
    """biased_sample."""
    p = np.array(scores)
    p = np.nan_to_num(p)
    p[p < 0] = 0
    # if there are non zero probabilities
    # replace negative or zero probabilities with smallest positive prob
    if len(p[p > 0]) > 1:
        min_p = np.min(p[p > 0])
        p[p == 0] = min_p
        p = p / p.sum()
        sample_ids = np.random.choice(
            len(graphs), size=sample_size, replace=False, p=p)
    # otherwise sample uniformly at random
    else:
        sample_ids = np.random.choice(
            len(graphs), size=sample_size, replace=False)
    sample_graphs = [graphs[sample_id] for sample_id in sample_ids]
    sample_scores = [scores[sample_id] for sample_id in sample_ids]
    return sample_graphs, sample_scores


def tournament_sample(graphs, scores, sample_size):
    n = len(graphs)
    ids = list(range(n))
    random.shuffle(ids)
    policy_selected_graphs = []
    policy_selected_scores = []
    for i in range(0, n - 1, 2):
        if scores[ids[i]] > scores[ids[i + 1]]:
            policy_selected_graphs.append(graphs[ids[i]])
            policy_selected_scores.append(scores[ids[i]])
        else:
            policy_selected_graphs.append(graphs[ids[i + 1]])
            policy_selected_scores.append(scores[ids[i + 1]])
    policy_selected_graphs, policy_selected_scores = policy_selected_graphs[:sample_size], policy_selected_scores[:sample_size]
    return policy_selected_graphs, policy_selected_scores


def sample(graphs, scores, sample_size, greedy_frac=0.5, policy='tournament'):
    """sample."""
    assert len(graphs) > 0, 'Something went wrong: no graphs left to sample' 
    if sample_size > len(scores):
        sample_size = len(scores)

    # select with greedy strategy
    greedy_selected_graphs = []
    greedy_selected_scores = []
    n_greedy_selection = int(sample_size * greedy_frac)
    sorted_ids = np.argsort(-np.array(scores))
    greedy_selected_ids = sorted_ids[:n_greedy_selection]
    if len(greedy_selected_ids) > 0:
        greedy_selected_graphs = [graphs[greedy_selected_id]
                                  for greedy_selected_id in greedy_selected_ids]
        greedy_selected_scores = [scores[greedy_selected_id]
                                  for greedy_selected_id in greedy_selected_ids]
    # select the rest with a specific policy: either biased or random
    policy_selected_graphs = []
    policy_selected_scores = []
    unselected_ids = sorted_ids[n_greedy_selection:]
    if len(unselected_ids) > 0:
        unselected_graphs = [graphs[unselected_id]
                             for unselected_id in unselected_ids]
        unselected_scores = [scores[unselected_id]
                             for unselected_id in unselected_ids]
        size = sample_size - len(greedy_selected_graphs)
        if policy == 'tournament':
            policy_selected_graphs, policy_selected_scores = tournament_sample(
                unselected_graphs, unselected_scores, size)
        elif policy == 'biased':
            policy_selected_graphs, policy_selected_scores = biased_sample(
                unselected_graphs, unselected_scores, size)

    selected_graphs = greedy_selected_graphs + policy_selected_graphs
    selected_scores = greedy_selected_scores + policy_selected_scores
    return selected_graphs, selected_scores


@curry
def _perturb(g, neighborhood_estimator=None, part_importance_estimator=None, feasibility_estimator=None):
    neighbor_graphs = []
    neighborhood_estimator.part_importance_estimator = part_importance_estimator
    neighbors = neighborhood_estimator.neighbors(g)
    if feasibility_estimator is None:
        feasibility_scores = [True] * len(neighbors)
    else:
        feasibility_scores = feasibility_estimator.predict(neighbors)
    for neighbor, is_feasible in zip(neighbors, feasibility_scores):
        if is_feasible:
            neighbor.graph['parent'] = g.copy()
            # TODO if the score is the same, then evaluate if they are the same, if so then use 'identity' as type
            neighbor.graph['type'] = type(neighborhood_estimator).__name__
            neighbor_graphs.append(neighbor)
    return neighbor_graphs


def perturb(graphs, neighborhood_estimator, part_importance_estimator=None, feasibility_estimator=None, execute_concurrently=False):
    """perturb."""
    # generate a fixed num of neighbors for each graph in input
    _perturb_ = _perturb(neighborhood_estimator=neighborhood_estimator, part_importance_estimator=part_importance_estimator, feasibility_estimator=feasibility_estimator)
    if execute_concurrently:
        neighbors_list = simple_parallel_map(_perturb_, graphs)
    else:
        neighbors_list = [_perturb_(g) for g in graphs]
    neighbor_graphs = []
    for neighbors in neighbors_list:
        neighbor_graphs.extend(neighbors)
    return neighbor_graphs


def elitism(proposed_graphs, oracle_func, frac_instances_to_remove_per_iter):
    """elitism."""
    scores = [oracle_func(g) for g in proposed_graphs]
    ids = np.argsort(scores)
    n = int(frac_instances_to_remove_per_iter * len(scores))
    ids = ids[n:]
    surviving_graphs = [proposed_graphs[id] for id in ids]
    surviving_scores = [scores[id] for id in ids]
    for graph, score in zip(surviving_graphs, surviving_scores):
        graph.graph['oracle_score'] = score
    return surviving_graphs, surviving_scores


@curry
def materialize_iterated_neighborhood(
        id_estimator,
        neighborhood_estimators=None,
        next_proposed_graphs=None,
        graphs=None,
        part_importance_estimator=None,
        feasibility_estimator=None,
        score_estimator=None,
        step=None,
        n_steps_driven_by_estimator=None,
        sample_size_to_perturb=None,
        n_queries_to_oracle_per_iter=None,
        parallelization_strategy=None,
        greedy_frac=0.5):
    """materialize_iterated_neighborhood."""
    if parallelization_strategy == 'graph_wise':
        execute_concurrently = True
    else:
        execute_concurrently = False
    neighborhood_estimator = neighborhood_estimators[id_estimator]
    start_time = time.process_time()
    all_neighbor_graphs = perturb(
        next_proposed_graphs,
        neighborhood_estimator,
        part_importance_estimator,
        feasibility_estimator,
        execute_concurrently)
    next_proposed_graphs, num_estimator_queries = [], 0
    neighbor_graphs = remove_duplicates(all_neighbor_graphs)
    if len(neighbor_graphs) == 0:
        logger.debug('Warning: removing duplicates among results resulted in no graphs')
        return next_proposed_graphs, num_estimator_queries
    neighbor_graphs = remove_duplicates_in_set(neighbor_graphs, graphs)
    if len(neighbor_graphs) == 0:
        logger.debug('Warning: removing duplicates w.r.t. archived graphs resulted in no graphs')
        return next_proposed_graphs, num_estimator_queries
    predicted_scores = score_estimator.acquisition_score(neighbor_graphs)
    num_estimator_queries = len(predicted_scores)
    # sample neighborhood according to surrogate score
    if step < n_steps_driven_by_estimator - 1:
        sample_size = int(sample_size_to_perturb / len(neighborhood_estimators))
    else:
        sample_size = int(n_queries_to_oracle_per_iter / len(neighborhood_estimators))
    sample_size = max(2, sample_size)
    next_proposed_graphs, next_proposed_scores = sample(
        neighbor_graphs,
        predicted_scores,
        sample_size,
        greedy_frac=greedy_frac)
    if len(next_proposed_graphs) == 0:
        logger.debug('Warning: sampling graphs resulted in no graphs')
        return next_proposed_graphs, num_estimator_queries
    
    end_time = time.process_time()
    elapsed_time = (end_time - start_time) / 60.0
    if n_steps_driven_by_estimator == 1:
        step_str = ''
    else:
        step_str = '%d:' % (step + 1)
    logger.info('%s%2d/%d) %30s:%4d novel graphs out of %4d generated  %3d selected graphs   best predicted score:%.3f   time:%.1f min' % (
        step_str, id_estimator + 1, len(neighborhood_estimators),
        type(neighborhood_estimator).__name__.replace('Neighborhood', ''),
        len(neighbor_graphs), len(all_neighbor_graphs),
        len(next_proposed_graphs), max(next_proposed_scores),
        elapsed_time))
    return next_proposed_graphs, num_estimator_queries


def select_iterated_neighborhoods(
        proposed_graphs,
        neighborhood_fitting_graphs,
        neighborhood_fitting_scores,
        graphs,
        neighborhood_estimators,
        part_importance_estimator,
        feasibility_estimator,
        score_estimator,
        n_steps_driven_by_estimator,
        sample_size_to_perturb,
        n_queries_to_oracle_per_iter,
        parallelization_strategy,
        greedy_frac):
    """select_iterated_neighborhoods."""
    all_n_estimator_queries = 0
    if parallelization_strategy == 'neighborhood_wise':
        execute_concurrently = True
    else:
        execute_concurrently = False
    for n_estimator, neighborhood_estimator in enumerate(neighborhood_estimators):
        neighborhood_estimator.fit(neighborhood_fitting_graphs, neighborhood_fitting_scores)
    for step in range(n_steps_driven_by_estimator):
        all_proposed_graphs = []
        _materialize_iterated_neighborhood_ = materialize_iterated_neighborhood(
            neighborhood_estimators=neighborhood_estimators,
            next_proposed_graphs=proposed_graphs[:],
            graphs=graphs,
            part_importance_estimator=part_importance_estimator,
            feasibility_estimator=feasibility_estimator,
            score_estimator=score_estimator,
            step=step,
            n_steps_driven_by_estimator=n_steps_driven_by_estimator,
            sample_size_to_perturb=sample_size_to_perturb,
            n_queries_to_oracle_per_iter=n_queries_to_oracle_per_iter)
        if execute_concurrently:
            list_of_graphs = simple_parallel_map(_materialize_iterated_neighborhood_, range(len(neighborhood_estimators)))
        else:
            list_of_graphs = [_materialize_iterated_neighborhood_(i) for i in range(len(neighborhood_estimators))]
        for gs, n_estimator_queries in list_of_graphs:
            all_proposed_graphs += gs
            all_n_estimator_queries += n_estimator_queries
        # for n_estimator, neighborhood_estimator in enumerate(neighborhood_estimators):
        #    next_proposed_graphs = _materialize_iterated_neighborhood_(n_estimator)
        #    all_proposed_graphs += next_proposed_graphs
        proposed_graphs = remove_duplicates(all_proposed_graphs)
        proposed_graphs = remove_duplicates_in_set(proposed_graphs, graphs)
        proposed_predicted_scores = score_estimator.acquisition_score(proposed_graphs)
        if step < n_steps_driven_by_estimator - 1:
            sample_size = sample_size_to_perturb
        else:
            sample_size = n_queries_to_oracle_per_iter
        proposed_graphs, proposed_predicted_scores = sample(
            proposed_graphs,
            proposed_predicted_scores,
            sample_size,
            greedy_frac)

    if n_queries_to_oracle_per_iter < len(proposed_graphs):
        logger.info('sampling %d out of %d non redundant graphs out of %d graphs generated for oracle evaluation' % (
            n_queries_to_oracle_per_iter, len(proposed_graphs), len(all_proposed_graphs)))
        proposed_graphs, proposed_predicted_scores = sample(
            proposed_graphs,
            proposed_predicted_scores,
            n_queries_to_oracle_per_iter,
            greedy_frac)
    # at this point we have proposed_graphs, proposed_scores:
    # for each graph we have the 'parent' and the 'type' and the score
    # we can formulate a learning task where starting from the parent graph
    # we have to predict in multiclass the type that has max score in any of the offsprings
    # so we have to collect all predictions relative to the same (hashed) parent graph
    # and select the argmax as the class to predict
    # the prediction can be used to allocate resources: expand only the k types that are
    # predicted to be yielding the best future improvements with k depending on the budget
    # the type predictor will be passed to the 'perturb' function and will mute the generation
    # for types predicted to under perform
    # NOTE: code should allow empty neighbor_graphs
    # Note: allow for no prediction at all when all types need to be generated

    return proposed_graphs, proposed_predicted_scores, all_n_estimator_queries


def compute_prediction_correlation(predicted_scores, true_scores):
    val, p = sp.stats.spearmanr(predicted_scores, true_scores)
    if np.isnan(val):
        val = 0
    return val


def optimize(graphs,
             oracle_func=None,
             n_iter=100,
             n_queries_to_oracle_per_iter=100,
             frac_instances_to_remove_per_iter=.1,
             sample_size_to_perturb=8,
             neighborhood_estimators=None,
             score_estimator=None,
             part_importance_estimator=None,
             feasibility_estimator=None,
             threshold_score_to_terminate=.99,
             monitor=None,
             draw_graphs=None,
             parallelization_strategy='graph_wise',
             greedy_sample_vs_tournament_frac=0.5):
    """optimize."""
    assert oracle_func is not None, 'Oracle must be available'
    n_steps_driven_by_estimator = 1
    original_n_queries_to_oracle_per_iter = n_queries_to_oracle_per_iter
    exploration_vs_exploitation = score_estimator.exploration_vs_exploitation
    oracle_start_time = time.process_time()
    true_scores = [oracle_func(graph)
                   if graph.graph.get('oracle_score', None) is None
                   else graph.graph['oracle_score']
                   for graph in graphs]
    for graph, score in zip(graphs, true_scores):
        graph.graph['oracle_score'] = score
    oracle_end_time = time.process_time()
    oracle_elapsed_time = (oracle_end_time - oracle_start_time) / 60.0
    logger.info('Oracle evaluated on %d graphs in %.1f min' % (len(graphs), oracle_elapsed_time))

    proposed_graphs = []
    proposed_true_scores = []
    proposed_predicted_scores = []
    corr_true_vs_pred_scores_history = []
    max_score_history = []
    corr_true_vs_pred_scores_delta = 0
    corr_true_vs_pred_scores = 0
    max_score_delta = 0
    for i in range(n_iter):
        logger.info('\n\n- iteration: %d/%d' % (i + 1, n_iter))

        # update with oracle
        num_proposed_graphs = len(proposed_graphs)
        if num_proposed_graphs:
            oracle_start_time = time.process_time()
            proposed_graphs, proposed_true_scores = elitism(
                proposed_graphs, oracle_func, frac_instances_to_remove_per_iter)
            graphs += proposed_graphs[:]
            true_scores += proposed_true_scores[:]
            weights, errors = score_estimator.estimate_weights(proposed_graphs, proposed_true_scores)
            estimator_names = [type(estimator).__name__.replace('Graph', '').replace('Score', '').replace('Estimator', '') for estimator in score_estimator.estimators]
            for es, er, we in zip(estimator_names, errors, weights):
                print('est: %30s:   err: %.1e   w:%.3f' % (es, er, we))
            selected_proposed_predicted_scores = score_estimator.predict(proposed_graphs)
            if len(proposed_true_scores) == len(selected_proposed_predicted_scores):
                corr_true_vs_pred_scores = compute_prediction_correlation(proposed_true_scores, selected_proposed_predicted_scores)
            else:
                corr_true_vs_pred_scores = 0
            corr_true_vs_pred_scores_history.append(corr_true_vs_pred_scores)
            if len(corr_true_vs_pred_scores_history) > 1:
                corr_true_vs_pred_scores_delta = (corr_true_vs_pred_scores_history[-1] - corr_true_vs_pred_scores_history[-2])
            else:
                corr_true_vs_pred_scores_delta = 0
            oracle_end_time = time.process_time()
            oracle_elapsed_time = (oracle_end_time - oracle_start_time) / 60.0
            logger.info('Oracle evaluated on %d graphs in %.1f min' % (num_proposed_graphs, oracle_elapsed_time))
            logger.info('Correlation between %d predicted and true scores: %.3f (increased of %.3f from previous iteration)' %
                        (len(proposed_true_scores), corr_true_vs_pred_scores, corr_true_vs_pred_scores_delta))

        if corr_true_vs_pred_scores_delta > 0:
            n_steps_driven_by_estimator_factor = 1
        else:
            n_steps_driven_by_estimator_factor = -1
        n_steps_driven_by_estimator += n_steps_driven_by_estimator_factor
        n_steps_driven_by_estimator = int(n_steps_driven_by_estimator)
        if corr_true_vs_pred_scores < 0.5:
            n_steps_driven_by_estimator = 1
        n_steps_driven_by_estimator = max(1, n_steps_driven_by_estimator)
        n_steps_driven_by_estimator = min(10, n_steps_driven_by_estimator)
        logger.info('n_steps_driven_by_estimator: %d' % (n_steps_driven_by_estimator))
        # evaluation of termination condition and current status output
        if proposed_true_scores:
            max_iteration_score = max(proposed_true_scores)
        else:
            max_iteration_score = 0
        max_score = max(true_scores)
        max_score_history.append(max_score)
        if max_score >= threshold_score_to_terminate:
            logger.info('Termination! score:%.3f is above user defined threshold:%.3f' % (max_score, threshold_score_to_terminate))
            break
        else:
            logger.info('Max score in last iteration: %.3f    Global max score: %.3f' % (max_iteration_score, max_score))
            if draw_graphs is not None:
                if proposed_graphs:
                    logger.info('Current iteration')
                    draw_graphs(proposed_graphs)
                logger.info('Current status')
                draw_graphs(graphs)
        if len(max_score_history) > 1:
            max_score_delta = max_score_history[-1] - max_score_history[-2]
            if max_score_delta <= 0:
                logger.info('Lack of improvement detected')
                n_queries_to_oracle_per_iter_increment_factor = 1.2
                exploration_vs_exploitation_factor = 10
            else:
                n_queries_to_oracle_per_iter_increment_factor = 0.08
                exploration_vs_exploitation_factor = 0.1
        else:
            max_score_delta = 0
            n_queries_to_oracle_per_iter_increment_factor = 1
            exploration_vs_exploitation_factor = 1
        exploration_vs_exploitation *= exploration_vs_exploitation_factor
        exploration_vs_exploitation = min(1, exploration_vs_exploitation)
        exploration_vs_exploitation = max(1e-6, exploration_vs_exploitation)
        score_estimator.set_exploration_vs_exploitation(exploration_vs_exploitation)
        logger.info('exploration_vs_exploitation:%.2e' % exploration_vs_exploitation)
        n_queries_to_oracle_per_iter *= n_queries_to_oracle_per_iter_increment_factor
        n_queries_to_oracle_per_iter = int(n_queries_to_oracle_per_iter)
        n_queries_to_oracle_per_iter = max(original_n_queries_to_oracle_per_iter, n_queries_to_oracle_per_iter)
        n_queries_to_oracle_per_iter = min(original_n_queries_to_oracle_per_iter * 20, n_queries_to_oracle_per_iter)
        logger.info('n_queries_to_oracle_per_iter:%d' % n_queries_to_oracle_per_iter)

        # update score_estimator
        score_estimator_start_time = time.process_time()
        score_estimator.fit(graphs, true_scores)
        score_estimator_end_time = time.process_time()
        score_estimator_elapsed_time = (score_estimator_end_time - score_estimator_start_time) / 60.0
        logger.info('Score estimator fitted on %d graphs in %.1f min' % (len(graphs), score_estimator_elapsed_time))

        # update part_importance_estimator
        part_importance_estimator_start_time = time.process_time()
        part_importance_estimator.fit(graphs, true_scores)
        part_importance_estimator_end_time = time.process_time()
        part_importance_estimator_elapsed_time = (part_importance_estimator_end_time - part_importance_estimator_start_time) / 60.0
        logger.info('Part importance  estimator fitted on %d graphs in %.1f min' % (len(graphs), part_importance_estimator_elapsed_time))

        # select small number (sample_size_to_perturb) of promising graphs for neighborhood expansion
        proposed_graphs, proposed_true_scores = sample(
            graphs,
            true_scores,
            sample_size_to_perturb,
            greedy_sample_vs_tournament_frac)

        # materialize neighborhood and select best candidates using score estimator
        iter_start_time = time.process_time()
        logger.info('From a biased draw of %d samples from %d graphs...' % (sample_size_to_perturb, len(graphs)))
        proposed_graphs, proposed_predicted_scores, all_n_estimator_queries = select_iterated_neighborhoods(
            proposed_graphs,
            graphs,
            true_scores,
            graphs,
            neighborhood_estimators,
            part_importance_estimator,
            feasibility_estimator,
            score_estimator,
            n_steps_driven_by_estimator,
            sample_size_to_perturb,
            n_queries_to_oracle_per_iter,
            parallelization_strategy,
            greedy_sample_vs_tournament_frac)

        if monitor:
            monitor(i, proposed_graphs, graphs, score_estimator, all_n_estimator_queries)
        if len(proposed_graphs) == 0:
            break
        iter_end_time = time.process_time()
        iter_elapsed_time_m = (iter_end_time - iter_start_time) / 60.0
        iter_elapsed_time_h = iter_elapsed_time_m / 60.0
        logger.info('overall iteration time: %.1f min (%.1f h)' % (iter_elapsed_time_m, iter_elapsed_time_h))

    # final update with oracle
    num_proposed_graphs = len(proposed_graphs)
    if num_proposed_graphs:
        oracle_start_time = time.process_time()
        proposed_graphs, proposed_true_scores = elitism(
            proposed_graphs, oracle_func, frac_instances_to_remove_per_iter)
        max_iteration_score = max(proposed_true_scores)
        graphs += proposed_graphs[:]
        true_scores += proposed_true_scores[:]
        max_score = max(true_scores)
        oracle_end_time = time.process_time()
        oracle_elapsed_time = (oracle_end_time - oracle_start_time) / 60.0
        logger.info('Oracle evaluated on %d graphs in %.1f min' % (num_proposed_graphs, oracle_elapsed_time))
        logger.info('Max score in last iteration: %.3f' % (max_iteration_score))
        logger.info('Max score globally: %.3f' % (max_score))
        if draw_graphs is not None:
            if proposed_graphs:
                logger.info('Current iteration')
                draw_graphs(proposed_graphs)
            logger.info('Current status')
            draw_graphs(graphs)
    return graphs


def optimizer_setup(use_RandomForest_estimator=True,
                    use_Linear_estimator=False,
                    use_EI_estimator=False,
                    use_UCB_estimator=False,
                    use_KNN_stimator=False,
                    use_ANN_estimator=False,
                    ANN_estimator_hidden_layer_sizes=[100, 50],
                    n_estimators_ANN=10,
                    n_neighbors_KNN=5,
                    n_estimators=100,
                    exploration_vs_exploitation=0,
                    decomposition_score_estimator=None,
                    execute_estimator_concurrently=False,

                    use_feasibility_estimator=True,
                    decomposition_feasibility_estimator=None,
                    domain_graphs_feasibility_estimator=None,

                    decomposition_part_importance_estimator=None,

                    use_part_importance_graph_grammar=False,
                    n_neighbors_part_importance_graph_grammar=None,
                    conservativeness_part_importance_graph_grammar=1,
                    max_num_substitutions_part_importance_graph_grammar=None,
                    context_size_part_importance_graph_grammar=1,
                    decomposition_part_importance_graph_grammar=None,
                    domain_graphs_part_importance_graph_grammar=None,
                    fit_at_each_iteration_part_importance_graph_grammar=False,

                    use_adaptive_graph_grammar=False,
                    n_neighbors_adaptive_graph_grammar=None,
                    conservativeness_adaptive_graph_grammar=1,
                    max_num_substitutions_adaptive_graph_grammar=None,
                    context_size_adaptive_graph_grammar=1,
                    part_size_adaptive_graph_grammar=4,
                    decomposition_adaptive_graph_grammar_base=None,
                    decomposition_adaptive_graph_grammar_approx=None,


                    use_edge_crossover=False,
                    n_neighbors_edge_crossover=10,
                    n_edge_edge_crossover=1,
                    tournament_size_factor_edge_crossover=10,
                    size_n_std_to_accept_edge_crossover=2,

                    use_cycle_crossover=False,
                    n_neighbors_cycle_crossover=10,
                    n_cycle_cycle_crossover=1,
                    n_edges_per_cycle_cycle_crossover=1,
                    tournament_size_factor_cycle_crossover=10,
                    size_n_std_to_accept_cycle_crossover=2,

                    use_double_edge_crossover=False,
                    n_neighbors_double_edge_crossover=10,
                    n_double_edges_double_edge_crossover=10,
                    tournament_size_factor_double_edge_crossover=10,
                    size_n_std_to_accept_double_edge_crossover=2,

                    use_edge_swapping=False,
                    n_neighbors_edge_swapping=None, n_edge_swapping=1,

                    use_edge_label_swapping=False,
                    n_neighbors_edge_label_swapping=None,
                    n_edge_label_swapping=1,

                    use_edge_label_mutation=False,
                    n_neighbors_edge_mutation=None, n_edge_mutation=1,

                    use_edge_move=False,
                    n_neighbors_edge_move=None, n_edge_move=1,

                    use_edge_removal=False,
                    n_neighbors_edge_removal=None, n_edge_removal=1,

                    use_edge_addition=False,
                    n_neighbors_edge_addition=None, n_edge_addition=1,

                    use_edge_expand=False,
                    n_neighbors_edge_expand=None, n_edge_expand=1,

                    use_edge_contract=False,
                    n_neighbors_edge_contract=None, n_edge_contract=1,


                    use_node_label_swapping=False,
                    n_neighbors_node_label_swapping=None,
                    n_node_label_swapping=1,

                    use_node_label_mutation=False,
                    n_neighbors_node_mutation=None, n_node_mutation=1,

                    use_node_removal=False,
                    n_neighbors_node_removal=None, n_node_removal=1,

                    use_node_addition=False,
                    n_neighbors_node_addition=None, n_node_addition=1,

                    use_node_smooth=False,
                    n_neighbors_node_smooth=None, n_node_smooth=1):
    """optimizer_setup."""
    part_importance_estimator = PartImportanceEstimator(
        decompose_func=decomposition_part_importance_estimator)

    neighborhood_estimators = []

    if use_edge_crossover:
        nec = NeighborhoodEdgeCrossover(
            n_edges=n_edge_edge_crossover,
            n_neighbors=n_neighbors_edge_crossover,
            tournament_size_factor=tournament_size_factor_edge_crossover,
            size_n_std_to_accept=size_n_std_to_accept_edge_crossover)
        neighborhood_estimators.append(nec)

    if use_cycle_crossover:
        ncc = NeighborhoodCycleCrossover(
            n_cycles=n_cycle_cycle_crossover,
            n_edges_per_cycle=n_edges_per_cycle_cycle_crossover,
            n_neighbors=n_neighbors_cycle_crossover,
            tournament_size_factor=tournament_size_factor_cycle_crossover,
            size_n_std_to_accept=size_n_std_to_accept_cycle_crossover)
        neighborhood_estimators.append(ncc)

    if use_double_edge_crossover:
        ndec = NeighborhoodDoubleEdgeCrossover(
            n_double_edges=n_double_edges_double_edge_crossover,
            n_neighbors=n_neighbors_double_edge_crossover,
            tournament_size_factor=tournament_size_factor_double_edge_crossover,
            size_n_std_to_accept=size_n_std_to_accept_double_edge_crossover)
        neighborhood_estimators.append(ndec)

    if use_edge_swapping:
        nes = NeighborhoodEdgeSwap(
            n_edges=n_edge_swapping, n_neighbors=n_neighbors_edge_swapping)
        neighborhood_estimators.append(nes)

    if use_edge_label_swapping:
        nels = NeighborhoodEdgeLabelSwap(
            n_edges=n_edge_label_swapping, n_neighbors=n_neighbors_edge_label_swapping)
        neighborhood_estimators.append(nels)

    if use_edge_label_mutation:
        nelm = NeighborhoodEdgeLabelMutation(
            n_edges=n_edge_mutation, n_neighbors=n_neighbors_edge_mutation)
        neighborhood_estimators.append(nelm)

    if use_edge_move:
        nem = NeighborhoodEdgeMove(
            n_edges=n_edge_move, n_neighbors=n_neighbors_edge_move)
        neighborhood_estimators.append(nem)

    if use_edge_removal:
        ner = NeighborhoodEdgeRemove(
            n_edges=n_edge_removal, n_neighbors=n_neighbors_edge_removal)
        neighborhood_estimators.append(ner)

    if use_edge_addition:
        nea = NeighborhoodEdgeAdd(
            n_edges=n_edge_addition, n_neighbors=n_neighbors_edge_addition)
        neighborhood_estimators.append(nea)

    if use_edge_expand:
        nee = NeighborhoodEdgeExpand(
            n_edges=n_edge_expand, n_neighbors=n_neighbors_edge_expand)
        neighborhood_estimators.append(nee)

    if use_edge_contract:
        nec = NeighborhoodEdgeContract(
            n_edges=n_edge_contract, n_neighbors=n_neighbors_edge_contract)
        neighborhood_estimators.append(nec)

    if use_node_label_swapping:
        nnls = NeighborhoodNodeLabelSwap(
            n_nodes=n_node_label_swapping,
            n_neighbors=n_neighbors_node_label_swapping)
        neighborhood_estimators.append(nnls)

    if use_node_label_mutation:
        nnlm = NeighborhoodNodeLabelMutation(
            n_nodes=n_node_mutation, n_neighbors=n_neighbors_node_mutation)
        neighborhood_estimators.append(nnlm)

    if use_node_removal:
        nnr = NeighborhoodNodeRemove(
            n_nodes=n_node_removal, n_neighbors=n_neighbors_node_removal)
        neighborhood_estimators.append(nnr)

    if use_node_addition:
        nna = NeighborhoodNodeAdd(
            n_nodes=n_node_addition, n_neighbors=n_neighbors_node_addition)
        neighborhood_estimators.append(nna)

    if use_node_smooth:
        nns = NeighborhoodNodeSmooth(
            n_nodes=n_node_smooth, n_neighbors=n_neighbors_node_smooth)
        neighborhood_estimators.append(nns)

    if use_part_importance_graph_grammar:
        npigge = NeighborhoodPartImportanceGraphGrammar(
            decomposition_function=decomposition_part_importance_graph_grammar,
            context=context_size_part_importance_graph_grammar,
            count=conservativeness_part_importance_graph_grammar,
            filter_max_num_substitutions=max_num_substitutions_part_importance_graph_grammar,
            frac_nodes_to_select=.5,
            n_neighbors=n_neighbors_part_importance_graph_grammar,
            fit_at_each_iteration=fit_at_each_iteration_part_importance_graph_grammar,
            domain_graphs=domain_graphs_part_importance_graph_grammar)
        # if fitting is not done iteratively then fit once on the domain_graphs here
        if fit_at_each_iteration_part_importance_graph_grammar is False:
            # fitting the grammar always adds the input graphs to the domain_graphs
            # so here we will add no novel graphs to the domain_graphs
            npigge.fit_grammar([])
        neighborhood_estimators.append(npigge)

    if use_adaptive_graph_grammar:
        ane = NeighborhoodAdaptiveGraphGrammar(
            base_decomposition_function=decomposition_adaptive_graph_grammar_base,
            approximate_decomposition_function=decomposition_adaptive_graph_grammar_approx,
            context=context_size_adaptive_graph_grammar,
            count=conservativeness_adaptive_graph_grammar,
            filter_max_num_substitutions=max_num_substitutions_adaptive_graph_grammar,
            n_neighbors=n_neighbors_adaptive_graph_grammar,
            ktop=part_size_adaptive_graph_grammar,
            enforce_connected=True)
        neighborhood_estimators.append(ane)

    score_estimators = []
    if use_UCB_estimator:
        score_estimator = GraphUpperConfidenceBoundEstimator(
            decomposition_funcs=decomposition_score_estimator,
            exploration_vs_exploitation=exploration_vs_exploitation)
        score_estimators.append(score_estimator)

    if use_RandomForest_estimator:
        score_estimator = GraphRandomForestScoreEstimator(
            decomposition_funcs=decomposition_score_estimator,
            n_estimators=n_estimators,
            exploration_vs_exploitation=exploration_vs_exploitation)
        score_estimators.append(score_estimator)

    if use_Linear_estimator:
        score_estimator = GraphLinearScoreEstimator(
            decomposition_funcs=decomposition_score_estimator,
            n_estimators=n_estimators,
            exploration_vs_exploitation=exploration_vs_exploitation)
        score_estimators.append(score_estimator)

    if use_EI_estimator:
        score_estimator = GraphExpectedImprovementEstimator(
            decomposition_funcs=decomposition_score_estimator,
            exploration_vs_exploitation=exploration_vs_exploitation)
        score_estimators.append(score_estimator)

    if use_ANN_estimator:
        score_estimator = GraphNeuralNetworkScoreEstimator(
            hidden_layer_sizes=ANN_estimator_hidden_layer_sizes,
            decomposition_funcs=decomposition_score_estimator,
            n_estimators=n_estimators_ANN,
            exploration_vs_exploitation=exploration_vs_exploitation)
        score_estimators.append(score_estimator)

    if use_KNN_stimator:
        score_estimator = GraphNearestNeighborScoreEstimator(
            n_neighbors=n_neighbors_KNN,
            decomposition_funcs=decomposition_score_estimator,
            n_estimators=n_estimators,
            exploration_vs_exploitation=exploration_vs_exploitation)
        score_estimators.append(score_estimator)

    score_estimator = EnsembleScoreEstimator(score_estimators, execute_concurrently=execute_estimator_concurrently)
    score_estimator.set_exploration_vs_exploitation(exploration_vs_exploitation)

    if use_feasibility_estimator:
        feasibility_estimator = FeasibilityEstimator(
            decomposition_funcs=decomposition_feasibility_estimator)
        logger.info('Fitting feasibility estimator on %d graphs' % len(domain_graphs_feasibility_estimator))
        feasibility_estimator.fit(domain_graphs_feasibility_estimator)
    else:
        feasibility_estimator = None
    return neighborhood_estimators, score_estimator, part_importance_estimator, feasibility_estimator
