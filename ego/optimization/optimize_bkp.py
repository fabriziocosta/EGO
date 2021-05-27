#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import random
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

from ego.optimization.score_estimator import GraphUpperConfidenceBoundEstimator
from ego.optimization.score_estimator import GraphRandomForestScoreEstimator
from ego.optimization.score_estimator import GraphLinearScoreEstimator
from ego.optimization.score_estimator import GraphExpectedImprovementEstimator
from ego.optimization.score_estimator import GraphNeuralNetworkScoreEstimator
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


def sample(graphs, scores, sample_size, greedy_frac=0.5, policy='random'):
    """sample."""
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
        if policy == 'random':
            ids = list(range(len(unselected_graphs)))
            random.shuffle(ids)
            policy_selected_graphs = [unselected_graphs[id] for id in ids[:size]]
            policy_selected_scores = [unselected_scores[id] for id in ids[:size]]
        elif policy == 'biased':
            policy_selected_graphs, policy_selected_scores = biased_sample(
                unselected_graphs, unselected_scores, size)

    selected_graphs = greedy_selected_graphs + policy_selected_graphs
    selected_scores = greedy_selected_scores + policy_selected_scores
    return selected_graphs, selected_scores


@curry
def _perturb(g, neighborhood_estimator=None, feasibility_estimator=None):
    neighbor_graphs = []
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


def perturb(graphs, neighborhood_estimator, feasibility_estimator=None, execute_concurrently=False):
    """perturb."""
    # generate a fixed num of neighbors for each graph in input
    _perturb_ = _perturb(neighborhood_estimator=neighborhood_estimator, feasibility_estimator=feasibility_estimator)
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
    surviving_graphs = [proposed_graphs[id] for id in ids[n:]]
    surviving_scores = [scores[id] for id in ids[n:]]
    for graph, score in zip(surviving_graphs, surviving_scores):
        graph.graph['oracle_score'] = score
    return surviving_graphs, surviving_scores


@curry
def materialize_iterated_neighborhood(
        n_estimator,
        neighborhood_estimators=None,
        next_proposed_graphs=None,
        graphs=None,
        feasibility_estimator=None,
        score_estimator=None,
        step=None,
        n_steps_driven_by_estimator=None,
        sample_size_to_perturb=None,
        n_queries_to_oracle_per_iter=None,
        parallelization_strategy=None):
    """materialize_iterated_neighborhood."""
    if parallelization_strategy == 'graph_wise':
        execute_concurrently = True
    else:
        execute_concurrently = False
    neighborhood_estimator = neighborhood_estimators[n_estimator]
    start_time = time.clock()
    all_neighbor_graphs = perturb(
        next_proposed_graphs,
        neighborhood_estimator,
        feasibility_estimator,
        execute_concurrently)
    neighbor_graphs = remove_duplicates(all_neighbor_graphs)
    neighbor_graphs = remove_duplicates_in_set(neighbor_graphs, graphs)
    predicted_scores = score_estimator.predict(neighbor_graphs)
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
        greedy_frac=0.5)
    end_time = time.clock()
    elapsed_time = (end_time - start_time) / 60.0
    if n_steps_driven_by_estimator == 1:
        step_str = ''
    else:
        step_str = '%d:' % (step + 1)
    logger.info('%s%2d/%d) %20s:%4d novel graphs out of %4d generated  %3d selected graphs   best predicted score:%.3f   time:%.1f min' % (
        step_str, n_estimator + 1, len(neighborhood_estimators),
        type(neighborhood_estimator).__name__.replace('Neighborhood', ''),
        len(neighbor_graphs), len(all_neighbor_graphs),
        len(next_proposed_graphs), max(next_proposed_scores),
        elapsed_time))
    return next_proposed_graphs


def select_iterated_neighborhoods(
        proposed_graphs,
        neighborhood_fitting_graphs,
        neighborhood_fitting_scores,
        graphs,
        neighborhood_estimators,
        feasibility_estimator,
        score_estimator,
        n_steps_driven_by_estimator,
        sample_size_to_perturb,
        n_queries_to_oracle_per_iter,
        parallelization_strategy):
    """select_iterated_neighborhoods."""
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
        for gs in list_of_graphs:
            all_proposed_graphs += gs
        # for n_estimator, neighborhood_estimator in enumerate(neighborhood_estimators):
        #    next_proposed_graphs = _materialize_iterated_neighborhood_(n_estimator)
        #    all_proposed_graphs += next_proposed_graphs
        proposed_graphs = remove_duplicates(all_proposed_graphs)
        proposed_graphs = remove_duplicates_in_set(proposed_graphs, graphs)
        predicted_scores = score_estimator.predict(proposed_graphs)
        if step < n_steps_driven_by_estimator - 1:
            sample_size = sample_size_to_perturb
        else:
            sample_size = n_queries_to_oracle_per_iter
        proposed_graphs, proposed_scores = sample(
            proposed_graphs,
            predicted_scores,
            sample_size,
            greedy_frac=0.5)

    if n_queries_to_oracle_per_iter < len(proposed_graphs):
        logger.info('sampling %d out of %d non redundant graphs out of %d graphs generated for oracle evaluation' % (
            n_queries_to_oracle_per_iter, len(proposed_graphs), len(all_proposed_graphs)))
        proposed_graphs, proposed_scores = sample(
            proposed_graphs,
            predicted_scores,
            n_queries_to_oracle_per_iter,
            greedy_frac=0.5)
    else:
        # keep all proposed graphs
        proposed_scores = predicted_scores
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
    return proposed_graphs, proposed_scores


def optimize(graphs,
             oracle_func=None,
             n_iter=100,
             budget_period_in_n_iter=4,
             n_queries_to_oracle_per_iter=100,
             frac_instances_to_remove_per_iter=.5,
             sample_size_to_perturb=25,
             n_steps_driven_by_estimator=4,
             increase_rate_of_steps_driven_by_estimator=1,
             sample_size_for_grammars=None,
             neighborhood_estimators=None,
             score_estimator=None,
             feasibility_estimator=None,
             threshold_score_to_terminate=.99,
             monitor=None,
             draw_graphs=None,
             parallelization_strategy='graph_wise'):
    """optimize."""
    assert oracle_func is not None, 'Oracle must be available'
    original_sample_size_to_perturb = sample_size_to_perturb
    original_n_queries_to_oracle_per_iter = n_queries_to_oracle_per_iter
    original_n_steps_driven_by_estimator = n_steps_driven_by_estimator
    current_n_steps_driven_by_estimator = 0
    oracle_start_time = time.clock()
    true_scores = [oracle_func(graph)
                   if graph.graph.get('oracle_score', None) is None
                   else graph.graph['oracle_score']
                   for graph in graphs]
    oracle_end_time = time.clock()
    oracle_elapsed_time = (oracle_end_time - oracle_start_time) / 60.0
    logger.info('Oracle evaluated on %d graphs in %.1f min' % (len(graphs), oracle_elapsed_time))

    for graph, score in zip(graphs, true_scores):
        graph.graph['oracle_score'] = score

    proposed_graphs = []
    proposed_scores = []
    for i in range(n_iter):
        # budget preparation
        budget_indicator = i % budget_period_in_n_iter + 1
        inverse_budget_indicator = budget_period_in_n_iter - i % budget_period_in_n_iter
        budget_fraction = budget_indicator / float(budget_period_in_n_iter)
        inverse_budget_fraction = inverse_budget_indicator / float(budget_period_in_n_iter)
        n_queries_to_oracle_per_iter = max(1, int(original_n_queries_to_oracle_per_iter * inverse_budget_fraction))
        sample_size_to_perturb = max(1, int(original_sample_size_to_perturb * inverse_budget_fraction))
        current_n_steps_driven_by_estimator = i * increase_rate_of_steps_driven_by_estimator
        n_steps_driven_by_estimator = max(1, int((original_n_steps_driven_by_estimator + current_n_steps_driven_by_estimator) * budget_fraction))
        logger.info('\n\n- iteration: %d' % (i + 1))
        logger.info('budget_indicator:%d   budget_fraction:%.2f   inverse_budget_indicator:%d   inverse_budget_fraction:%.2f   \nn_queries_to_oracle_per_iter:%d   sample_size_to_perturb:%d   n_steps_driven_by_estimator:%d' %
                    (budget_indicator, budget_fraction, inverse_budget_indicator, inverse_budget_fraction, n_queries_to_oracle_per_iter, sample_size_to_perturb, n_steps_driven_by_estimator))

        # update with oracle
        num_proposed_graphs = len(proposed_graphs)
        if num_proposed_graphs:
            oracle_start_time = time.clock()
            predicted_scores = proposed_scores
            proposed_graphs, proposed_scores = elitism(
                proposed_graphs, oracle_func, frac_instances_to_remove_per_iter)
            graphs += proposed_graphs[:]
            true_scores += proposed_scores[:]
            if len(predicted_scores) == len(proposed_scores):
                corr_true_vs_pred_scores = np.corrcoef(predicted_scores, proposed_scores)[0, 1]
            else:
                corr_true_vs_pred_scores = 0
            oracle_end_time = time.clock()
            oracle_elapsed_time = (oracle_end_time - oracle_start_time) / 60.0
            logger.info('Oracle evaluated on %d graphs in %.1f min' % (num_proposed_graphs, oracle_elapsed_time))
            logger.info('Correlation between predicted and true scores: %.3f' % (corr_true_vs_pred_scores))

        # evaluation of termination condition and current status output
        if proposed_scores:
            max_iteration_score = max(proposed_scores)
        else:
            max_iteration_score = 0
        max_score = max(true_scores)
        if max_score >= threshold_score_to_terminate:
            logger.info('Termination! score:%.3f is above user defined threshold:%.3f' % (max_score, threshold_score_to_terminate))
            break
        else:
            logger.info('Max score in last iteration: %.3f' % (max_iteration_score))
            logger.info('Max score globally: %.3f' % (max_score))
            if draw_graphs is not None:
                if proposed_graphs:
                    logger.info('Current iteration')
                    draw_graphs(proposed_graphs)
                logger.info('Current status')
                draw_graphs(graphs)

        # update score_estimator
        score_estimator_start_time = time.clock()
        score_estimator.fit(graphs, true_scores)
        score_estimator_end_time = time.clock()
        score_estimator_elapsed_time = (score_estimator_end_time - score_estimator_start_time) / 60.0
        logger.info('Score estimator fitted on %d graphs in %.1f min' % (len(graphs), score_estimator_elapsed_time))

        # update neighborhood_estimators
        if sample_size_for_grammars is None:
            neighborhood_fitting_graphs, neighborhood_fitting_scores = graphs, true_scores
        else:
            neighborhood_fitting_graphs, neighborhood_fitting_scores = sample(
                graphs,
                true_scores,
                sample_size_for_grammars,
                greedy_frac=0.5)

        # select small number (sample_size_to_perturb) of promising graphs for neighborhood expansion
        proposed_graphs, proposed_scores = sample(
            graphs,
            true_scores,
            sample_size_to_perturb,
            greedy_frac=0.5)

        # materialize neighborhood and select best candidates using score estimator
        iter_start_time = time.clock()
        logger.info('From a biased draw of %d samples from %d graphs...' % (sample_size_to_perturb, len(graphs)))
        proposed_graphs, proposed_scores = select_iterated_neighborhoods(
            proposed_graphs,
            neighborhood_fitting_graphs,
            neighborhood_fitting_scores,
            graphs,
            neighborhood_estimators,
            feasibility_estimator,
            score_estimator,
            n_steps_driven_by_estimator,
            sample_size_to_perturb,
            n_queries_to_oracle_per_iter,
            parallelization_strategy)

        if monitor:
            monitor(i, proposed_graphs, graphs, score_estimator)
        if len(proposed_graphs) == 0:
            break
        iter_end_time = time.clock()
        iter_elapsed_time_m = (iter_end_time - iter_start_time) / 60.0
        iter_elapsed_time_h = iter_elapsed_time_m / 60.0
        logger.info('overall iteration time: %.1f min (%.1f h)' % (iter_elapsed_time_m, iter_elapsed_time_h))

    # final update with oracle
    num_proposed_graphs = len(proposed_graphs)
    if num_proposed_graphs:
        oracle_start_time = time.clock()
        proposed_graphs, proposed_scores = elitism(
            proposed_graphs, oracle_func, frac_instances_to_remove_per_iter)
        max_iteration_score = max(proposed_scores)
        graphs += proposed_graphs[:]
        true_scores += proposed_scores[:]
        max_score = max(true_scores)
        oracle_end_time = time.clock()
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
                    use_NN_estimator=False,
                    NN_estimator_hidden_layer_sizes=[100, 50],
                    n_estimators_NN=7,
                    n_estimators=100,
                    exploration_vs_exploitation=0,
                    decomposition_score_estimator=None,

                    use_feasibility_estimator=True,
                    decomposition_feasibility_estimator=None,
                    domain_graphs_feasibility_estimator=None,

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
                    n_neighbors_edge_crossover=10, n_edge_edge_crossover=1,

                    use_cycle_crossover=False,
                    n_neighbors_cycle_crossover=10,
                    n_cycle_cycle_crossover=1,
                    n_edges_per_cycle_cycle_crossover=1,

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
    neighborhood_estimators = []

    if use_edge_crossover:
        nec = NeighborhoodEdgeCrossover(
            n_edges=n_edge_edge_crossover,
            n_neighbors=n_neighbors_edge_crossover)
        neighborhood_estimators.append(nec)

    if use_cycle_crossover:
        ncc = NeighborhoodCycleCrossover(
            n_cycles=n_cycle_cycle_crossover,
            n_edges_per_cycle=n_edges_per_cycle_cycle_crossover,
            n_neighbors=n_neighbors_cycle_crossover)
        neighborhood_estimators.append(ncc)

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

    if use_NN_estimator:
        score_estimator = GraphNeuralNetworkScoreEstimator(
            hidden_layer_sizes=NN_estimator_hidden_layer_sizes,
            decomposition_funcs=decomposition_score_estimator,
            n_estimators=n_estimators_NN,
            exploration_vs_exploitation=exploration_vs_exploitation)
        score_estimators.append(score_estimator)
    score_estimator = EnsembleScoreEstimator(score_estimators)

    if use_feasibility_estimator:
        feasibility_estimator = FeasibilityEstimator(
            decomposition_funcs=decomposition_feasibility_estimator)
        logger.info('Fitting feasibility estimator on %d graphs' % len(domain_graphs_feasibility_estimator))
        feasibility_estimator.fit(domain_graphs_feasibility_estimator)
    else:
        feasibility_estimator = None
    return neighborhood_estimators, score_estimator, feasibility_estimator
